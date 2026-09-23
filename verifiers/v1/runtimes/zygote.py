"""Warm fork server for Python programs started by the subprocess runtime.

A rollout starts several short-lived Python programs (an MCP tool server, a
harness chat program), and each fresh interpreter re-imports the same heavy
modules: two seconds or more per program for AutomationBench, multiplied under
concurrent rollouts. A zygote is one long-lived interpreter per Python
executable that imports those modules once and forks a child per program, so a
program starts with its imports already done.

A forked child is set up like a fresh ``python`` process: its own session
(``setsid``, so the runtime can signal the whole tree), the caller's working
directory, environment, and stdio descriptors, a fresh ``sys.argv`` and
``sys.path[0]``, and ``__main__`` execution through ``runpy``. Differences a
program could observe are that preloaded modules were imported under the
zygote's environment rather than its own and that children share the zygote's
string hash seed. Programs whose ``PYTHON*`` startup variables differ from the
zygote's, or whose command line uses other interpreter options, are exec'd
normally, as is everything when the zygote cannot start.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER = struct.Struct(">Q")
_SERVER_SOURCE = (Path(__file__).with_name("_zygote_server.py")).read_text()
_READY_TIMEOUT_S = 300.0
# Variables consumed at interpreter startup: a child cannot adopt new values, so
# a program whose values differ from the zygote's is exec'd instead. Stdio
# settings are applied per child.
_PER_CHILD_PYTHON_VARS = {"PYTHONUNBUFFERED", "PYTHONIOENCODING"}


def _startup_vars(env: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in env.items()
        if k.startswith("PYTHON") and k not in _PER_CHILD_PYTHON_VARS
    }


def same_interpreter(candidate: str, python: str) -> bool:
    """Whether `candidate` names `python`'s interpreter, e.g. `bin/python3` for
    `bin/python`. Venv interpreters symlink to a shared base binary, so the
    directory must match too: it is what selects the venv's site-packages."""
    if candidate == python:
        return True
    return os.path.dirname(os.path.abspath(candidate)) == os.path.dirname(
        os.path.abspath(python)
    ) and os.path.realpath(candidate) == os.path.realpath(python)


def eligible(argv: list[str], python: str) -> bool:
    """Whether `argv` is a plain `python script|-m module|-c code` invocation."""
    if len(argv) < 2 or not same_interpreter(argv[0], python):
        return False
    if argv[1] in ("-m", "-c"):
        return len(argv) >= 3
    return not argv[1].startswith("-")


class ZygoteProcess:
    """The subset of `asyncio.subprocess.Process` the runtime uses."""

    def __init__(
        self,
        pid: int,
        connection: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        stdin: asyncio.StreamWriter | None,
        stdout: asyncio.StreamReader | None,
        stderr: asyncio.StreamReader | None,
    ) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._connection = connection
        self._writer = writer
        self._exited = asyncio.ensure_future(self._watch())

    async def _watch(self) -> int:
        code = -signal.SIGKILL
        try:
            line = await self._connection.readline()
            if line:
                code = int(json.loads(line)["exit"])
            else:
                # The zygote died; make sure the program does not outlive it.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.pid, signal.SIGKILL)
        finally:
            self.returncode = code
            # Closing lets the zygote reap the child: the pid stays reserved
            # until returncode is visible, so signalling it never races reuse.
            self._writer.close()
        return code

    async def wait(self) -> int:
        return await asyncio.shield(self._exited)

    async def communicate(self) -> tuple[bytes, bytes]:
        async def drain(reader: asyncio.StreamReader | None) -> bytes:
            return b"" if reader is None else await reader.read()

        if self.stdin is not None:
            self.stdin.close()
        stdout, stderr = await asyncio.gather(drain(self.stdout), drain(self.stderr))
        await self.wait()
        return stdout, stderr


async def _read_pipe(fd: int) -> asyncio.StreamReader:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=2**16, loop=loop)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader, loop=loop),
        os.fdopen(fd, "rb", 0),
    )
    return reader


async def _write_pipe(fd: int) -> asyncio.StreamWriter:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(loop=loop),
        os.fdopen(fd, "wb", 0),
    )
    return asyncio.StreamWriter(transport, protocol, None, loop)


class Zygote:
    """One warm interpreter; `spawn` forks a program from it."""

    def __init__(self, python: str, preload: list[str], env: dict[str, str]) -> None:
        self.python = python
        self.preload = preload
        self.startup_vars = _startup_vars(env)
        self._dir = tempfile.mkdtemp(prefix="vf-zygote-")
        self.socket_path = os.path.join(self._dir, f"{uuid.uuid4().hex[:8]}.sock")
        self._process = subprocess.Popen(
            [python, "-c", _SERVER_SOURCE, self.socket_path, json.dumps(preload)],
            stdin=subprocess.PIPE,  # closed (EOF) when this worker exits
            stdout=subprocess.PIPE,
            env=env,
            cwd=self._dir,
            start_new_session=True,
        )
        assert self._process.stdout is not None
        ready = _readline(self._process.stdout, _READY_TIMEOUT_S)
        message = json.loads(ready) if ready else {"error": "exited during preload"}
        if "error" in message:
            self.close()
            raise RuntimeError(f"zygote for {python} failed: {message['error']}")
        for failure in message.get("failed", []):
            logger.warning("zygote for %s could not preload %s", python, failure)
        logger.info(
            "zygote for %s ready (preloaded %d modules, %s OS threads)",
            python,
            len(preload) - len(message.get("failed", [])),
            message.get("os_threads"),
        )

    def accepts(self, argv: list[str], env: dict[str, str]) -> bool:
        return (
            self._process.poll() is None
            and eligible(argv, self.python)
            and _startup_vars(env) == self.startup_vars
        )

    async def spawn(
        self,
        argv: list[str],
        env: dict[str, str],
        cwd: str,
        *,
        stdin: int | None,
        stdout: int | None,
        stderr: int | None,
    ) -> ZygoteProcess:
        """Fork `argv`. Each stdio argument is a descriptor to hand to the child,
        or None for a new pipe whose other end the returned process exposes."""
        opened: list[int] = []
        child_fds: list[int] = []
        parent_ends: list[int | None] = []
        try:
            for index, given in enumerate((stdin, stdout, stderr)):
                if given is not None:
                    child_fds.append(given)
                    parent_ends.append(None)
                    continue
                read_end, write_end = os.pipe()
                opened += [read_end, write_end]
                child, parent = (
                    (read_end, write_end) if index == 0 else (write_end, read_end)
                )
                child_fds.append(child)
                parent_ends.append(parent)
            body = json.dumps({"argv": argv, "env": env, "cwd": cwd}).encode()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.socket_path)
                socket.send_fds(sock, [_HEADER.pack(len(body)) + body], child_fds)
            except BaseException:
                sock.close()
                raise
            # The child holds its own copies now.
            for index, parent in enumerate(parent_ends):
                if parent is not None:
                    os.close(child_fds[index])
                    opened.remove(child_fds[index])
            sock.setblocking(False)
            reader, writer = await asyncio.open_unix_connection(sock=sock)
            reply = json.loads(await reader.readline() or b'{"error": "zygote closed"}')
            if "error" in reply:
                writer.close()
                raise RuntimeError(reply["error"])
            stdin_w = (
                await _write_pipe(parent_ends[0])
                if parent_ends[0] is not None
                else None
            )
            stdout_r = (
                await _read_pipe(parent_ends[1]) if parent_ends[1] is not None else None
            )
            stderr_r = (
                await _read_pipe(parent_ends[2]) if parent_ends[2] is not None else None
            )
            opened.clear()  # pipe ends now owned by transports
            return ZygoteProcess(
                reply["pid"], reader, writer, stdin_w, stdout_r, stderr_r
            )
        finally:
            for fd in opened:
                with contextlib.suppress(OSError):
                    os.close(fd)

    def close(self) -> None:
        if self._process.poll() is None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        with contextlib.suppress(OSError):
            os.unlink(self.socket_path)
        with contextlib.suppress(OSError):
            os.rmdir(self._dir)


def _readline(stream, timeout: float) -> bytes:
    result: list[bytes] = []
    thread = threading.Thread(
        target=lambda: result.append(stream.readline()), daemon=True
    )
    thread.start()
    thread.join(timeout)
    return result[0] if result else b""


_zygotes: dict[tuple, Zygote | None] = {}
_lock = threading.Lock()


def get_zygote(python: str, preload: list[str], env: dict[str, str]) -> Zygote | None:
    """The worker's zygote for `python`, started on first use; None if it failed."""
    key = (python, tuple(preload), tuple(sorted(_startup_vars(env).items())))
    with _lock:
        if key not in _zygotes:
            try:
                _zygotes[key] = Zygote(python, preload, env)
            except Exception as exc:  # noqa: BLE001 - any failure falls back to exec
                logger.warning("fork server disabled, using exec: %s", exc)
                _zygotes[key] = None
        return _zygotes[key]


def close_all() -> None:
    with _lock:
        for zygote in _zygotes.values():
            if zygote is not None:
                zygote.close()
        _zygotes.clear()
