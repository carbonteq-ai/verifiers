"""Fork server that starts Python programs from a warm, preloaded interpreter.

Not imported by Verifiers: `zygote.py` runs this file's source with
``python -c`` so that ``sys.path[0]`` is not this directory (which contains a
``subprocess.py`` that would shadow the standard library).

Protocol, one Unix-socket connection per program:

- request: an 8-byte big-endian length, then a JSON object ``{"argv", "env",
  "cwd"}``, sent together with three file descriptors (stdin, stdout, stderr)
  as ``SCM_RIGHTS`` ancillary data;
- replies: newline-delimited JSON, first ``{"pid": n}`` (or ``{"error": s}``),
  then ``{"exit": code}`` when the program exits, where ``code`` follows
  ``asyncio`` (negative signal number when killed by a signal).

The exited child stays unreaped (``waitid(WNOWAIT)``) until the client closes
its connection, so the client can signal the pid's process group until it has
observed the exit without racing pid reuse.
"""

import contextlib
import importlib
import json
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import traceback

_HEADER = struct.Struct(">Q")
RETIRED = "retired: a Python thread is running in the zygote"


def _read_request(conn):
    data, fds, _flags, _addr = socket.recv_fds(conn, 1 << 16, 3)
    while len(data) < _HEADER.size:
        chunk = conn.recv(1 << 16)
        if not chunk:
            raise ConnectionError("short request header")
        data += chunk
    (length,) = _HEADER.unpack_from(data)
    body = data[_HEADER.size :]
    while len(body) < length:
        chunk = conn.recv(1 << 16)
        if not chunk:
            raise ConnectionError("short request body")
        body += chunk
    return json.loads(body), fds


def _std_streams(env):
    import io

    unbuffered = bool(env.get("PYTHONUNBUFFERED"))
    encoding = sys.stdout.encoding
    errors = sys.stdout.errors
    io_encoding = env.get("PYTHONIOENCODING")
    if io_encoding:
        name, _, errs = io_encoding.partition(":")
        encoding = name or encoding
        errors = errs or errors

    def stream(fd, mode, *, stderr=False):
        interactive = os.isatty(fd)
        if mode == "r":
            raw = io.FileIO(fd, "r", closefd=False)
            buffered = raw if unbuffered else io.BufferedReader(raw)
            return io.TextIOWrapper(buffered, encoding=encoding, errors=errors)
        raw = io.FileIO(fd, "w", closefd=False)
        buffered = raw if unbuffered else io.BufferedWriter(raw)
        return io.TextIOWrapper(
            buffered,
            encoding=encoding,
            errors="backslashreplace" if stderr else errors,
            line_buffering=stderr or interactive,
            write_through=unbuffered,
        )

    sys.stdin = sys.__stdin__ = stream(0, "r")
    sys.stdout = sys.__stdout__ = stream(1, "w")
    sys.stderr = sys.__stderr__ = stream(2, "w", stderr=True)


def _exit_code(exc):
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _child(request, fds, close_fds, started):
    """Become the requested program. Never returns."""
    code = 1
    try:
        os.setsid()
        # Only now may the client learn the pid: it signals the process group,
        # which until setsid() is the zygote's own.
        os.write(started, b"s")
        os.close(started)
        signal.set_wakeup_fd(-1)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        for fd in close_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        for target, fd in enumerate(fds):
            os.dup2(fd, target)
        for fd in set(fds):
            if fd > 2:
                os.close(fd)
        os.chdir(request["cwd"])
        os.environ.clear()
        os.environ.update(request["env"])
        _std_streams(request["env"])

        import runpy
        import tempfile
        import warnings

        # Values some modules cache from the environment on first use; the
        # zygote's values must not leak into a program with its own env.
        tempfile.tempdir = None
        numpy = sys.modules.get("numpy.random")
        if numpy is not None:
            numpy.seed(None)
        # A preloaded target module is already in sys.modules; running it as
        # __main__ is still a fresh execution, so the runpy notice is noise.
        warnings.filterwarnings(
            "ignore",
            message=r".*found in sys\.modules after import of package.*",
            category=RuntimeWarning,
        )

        argv = request["argv"][1:]
        if argv[0] == "-m":
            sys.argv = [argv[1], *argv[2:]]
            sys.path[0] = os.getcwd()
            runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        elif argv[0] == "-c":
            sys.argv = ["-c", *argv[2:]]
            sys.path[0] = ""
            import types

            # A fresh module: this function's globals are the server's __main__.
            main = types.ModuleType("__main__")
            sys.modules["__main__"] = main
            exec(compile(argv[1], "<string>", "exec"), main.__dict__)  # noqa: S102 - `python -c`
        else:
            sys.argv = list(argv)
            sys.path[0] = os.path.dirname(os.path.abspath(argv[0]))
            runpy.run_path(argv[0], run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = _exit_code(exc)
    except BaseException:  # noqa: BLE001 - an uncaught error exits 1, as in python
        traceback.print_exc()
        code = 1
    # Interpreter shutdown, which ignores failures: the child must never return
    # into the server's loop.
    with contextlib.suppress(BaseException):
        threading._shutdown()  # join non-daemon threads
    with contextlib.suppress(BaseException):
        import atexit

        atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(BaseException):
            stream.flush()
    os._exit(code & 0xFF)


def _status(info):
    if info.si_code == os.CLD_EXITED:
        return info.si_status
    return -info.si_status


def main():
    sock_path = sys.argv[1]
    preload = json.loads(sys.argv[2])
    del sys.argv[1:]
    failed = []
    for name in preload:
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - reported; the module is skipped
            failed.append(f"{name}: {exc!r}")
    # Python-level threads hold interpreter locks a child could inherit
    # mid-update, so the zygote refuses them. Native pools some extensions start
    # at import (OpenBLAS, jemalloc) quiesce around fork through pthread_atfork,
    # which CPython's generic multi-threaded fork warning cannot see.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*use of fork\(\) may lead to deadlocks.*",
        category=DeprecationWarning,
    )
    if threading.active_count() > 1:
        names = [t.name for t in threading.enumerate()]
        print(json.dumps({"error": f"preload started threads {names}"}), flush=True)
        return
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    listener.listen(256)
    wake_r, wake_w = os.pipe()
    os.set_blocking(wake_r, False)
    os.set_blocking(wake_w, False)
    signal.set_wakeup_fd(wake_w)
    signal.signal(signal.SIGCHLD, lambda *_: None)
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ, "accept")
    selector.register(0, selectors.EVENT_READ, "control")
    selector.register(wake_r, selectors.EVENT_READ, "wake")
    seen_modules = set(preload)
    running = {}  # pid -> connection awaiting the exit report
    reported = {}  # connection -> pid exited but not yet reaped
    abandoned = set()  # pids whose client went away before they exited
    can_peek = hasattr(os, "waitid") and hasattr(os, "WNOWAIT")
    tasks = (
        len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task") else None
    )
    print(
        json.dumps({"ready": True, "failed": failed, "os_threads": tasks}), flush=True
    )
    # Output from later imports must not reach the client's ready pipe, nor sit
    # in a buffer a forked child would inherit and flush into its own stdout.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)

    def send(conn, message):
        try:
            conn.sendall(json.dumps(message).encode() + b"\n")
        except OSError:
            pass

    def drop(conn):
        try:
            selector.unregister(conn)
        except (KeyError, ValueError):
            pass
        conn.close()

    def reap():
        for pid, conn in list(running.items()):
            try:
                if can_peek:
                    info = os.waitid(
                        os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                    )
                    if info is None:
                        continue
                    status = _status(info)
                else:
                    done, raw = os.waitpid(pid, os.WNOHANG)
                    if done == 0:
                        continue
                    status = os.waitstatus_to_exitcode(raw)
            except ChildProcessError:
                status = -signal.SIGKILL
            del running[pid]
            if conn is None:
                abandoned.discard(pid)
                if can_peek:
                    os.waitpid(pid, 0)
                continue
            send(conn, {"exit": status})
            if can_peek:
                reported[conn] = pid

    while True:
        for key, _ in selector.select():
            tag = key.data
            if tag == "accept":
                conn, _ = listener.accept()
                try:
                    request, fds = _read_request(conn)
                except Exception as exc:  # noqa: BLE001 - reported to the client
                    send(conn, {"error": repr(exc)})
                    conn.close()
                    continue
                # A `-m` program's module is imported here on first use, so
                # later forks of it start warm without naming it in `preload`.
                argv = request["argv"]
                if argv[1:2] == ["-m"] and argv[2] not in seen_modules:
                    seen_modules.add(argv[2])
                    with contextlib.suppress(BaseException):
                        importlib.import_module(argv[2])
                if threading.active_count() > 1:
                    # An import started a Python thread: forking is no longer
                    # safe, so this zygote retires and callers exec instead.
                    send(conn, {"error": RETIRED})
                    conn.close()
                    for fd in fds:
                        os.close(fd)
                    continue
                close_fds = [listener.fileno(), wake_r, wake_w, conn.fileno()]
                close_fds.append(selector.fileno())
                close_fds += [c.fileno() for c in running.values() if c is not None]
                close_fds += [c.fileno() for c in reported]
                for stream in (sys.stdout, sys.stderr):
                    with contextlib.suppress(OSError, ValueError):
                        stream.flush()
                started_r, started_w = os.pipe()
                close_fds.append(started_r)
                pid = os.fork()
                if pid == 0:
                    _child(request, fds, close_fds, started_w)
                os.close(started_w)
                for fd in fds:
                    os.close(fd)
                # EOF without the byte means the child died before setsid();
                # its exit is still reported below.
                os.read(started_r, 1)
                os.close(started_r)
                running[pid] = conn
                send(conn, {"pid": pid})
                selector.register(conn, selectors.EVENT_READ, "client")
            elif tag == "control":
                if not os.read(0, 4096):
                    os.unlink(sock_path)
                    return
            elif tag == "wake":
                try:
                    while os.read(wake_r, 4096):
                        pass
                except BlockingIOError:
                    pass
                reap()
            else:
                conn = key.fileobj
                try:
                    data = conn.recv(4096)
                except OSError:
                    data = b""
                if data:
                    continue
                drop(conn)
                pid = reported.pop(conn, None)
                if pid is not None:
                    os.waitpid(pid, 0)
                    continue
                for pid, owner in list(running.items()):
                    if owner is conn:
                        running[pid] = None
                        abandoned.add(pid)
        # A SIGCHLD may coalesce with others; sweep on every wakeup.
        if running:
            reap()


main()
