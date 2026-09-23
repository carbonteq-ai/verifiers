"""Local subprocess runtime."""

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import stat
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, ClassVar, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import zygote
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
)
from verifiers.v1.utils.paths import CACHE_DIR

_BACKGROUND_STOP_TIMEOUT = 5

logger = logging.getLogger(__name__)

# Implicit host inheritance removes every name containing "API_KEY" while keeping
# harmless settings such as PATH, HOME, and cache locations. The explicit `env`
# argument is merged afterward, so callers can deliberately pass credentials and
# child processes inherit them. Containers and sandboxes inherit no host environment.


def _fork_server_default() -> bool:
    return os.environ.get("VF_FORK_SERVER", "").lower() in ("1", "true", "yes")


def _preload_default() -> list[str]:
    return [m for m in os.environ.get("VF_FORK_SERVER_PRELOAD", "").split(",") if m]


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"
    fork_server: bool = Field(default_factory=_fork_server_default)
    """Start Python programs that run on this worker's interpreter (tool servers,
    and harness scripts on a preinstalled interpreter) by forking a warm server
    that has already imported `preload`, instead of starting a new interpreter.
    See `verifiers.v1.runtimes.zygote` for what a forked program shares.
    Defaults to `VF_FORK_SERVER`, so one setting covers every subprocess runtime
    a worker creates, including tool servers' own runtimes."""
    preload: list[str] = Field(default_factory=_preload_default)
    """Modules the fork server imports once, e.g. a tool server's module and the
    harness program's client libraries. Defaults to the comma-separated
    `VF_FORK_SERVER_PRELOAD`."""


class SubprocessRuntimeInfo(SubprocessConfig, BaseRuntimeInfo):
    pass


async def read_stream(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while chunk := await reader.read(64 * 1024):
        yield chunk


def signal_process(
    process: "asyncio.subprocess.Process | zygote.ZygoteProcess",
    signal_: signal.Signals,
) -> None:
    if process.returncode is not None:
        return
    # open_process() creates a new session, so pgid == pid and signalling the
    # group reaps the complete process tree.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal_)


def _child_fd(stream: int | IO[bytes] | None, inherited: int) -> int | None:
    """The descriptor a forked child gets for an `asyncio` stdio argument;
    None means a new pipe."""
    if stream == asyncio.subprocess.PIPE:
        return None
    if stream is None:
        return inherited
    if isinstance(stream, int):
        return stream
    return stream.fileno()


class SubprocessProcess(RuntimeProcess):
    def __init__(
        self, process: "asyncio.subprocess.Process | zygote.ZygoteProcess"
    ) -> None:
        self._process = process
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdin = process.stdin
        self.stdout = read_stream(process.stdout)
        self.stderr = read_stream(process.stderr)

    async def write(self, data: bytes) -> None:
        self._stdin.write(data)
        await self._stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def terminate(self) -> None:
        signal_process(self._process, signal.SIGTERM)

    async def kill(self) -> None:
        signal_process(self._process, signal.SIGKILL)


class SubprocessRuntime(Runtime):
    # Share prepared script environments across the worker's per-rollout runtimes.
    scripts_dir: ClassVar[str] = str(CACHE_DIR / "runtimes" / "scripts")
    _interpreters: ClassVar[dict[str, str]] = {}
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(self, config: SubprocessConfig, name: str | None = None) -> None:
        super().__init__(name)
        self._uv_interpreters = self._interpreters
        self._uv_script_locks = self._locks
        self.config = config
        self.info = SubprocessRuntimeInfo(**config.model_dump())
        self.workdir: Path | None = None
        self._background: list[asyncio.subprocess.Process | zygote.ZygoteProcess] = []

    async def start(self) -> None:
        self.workdir = CACHE_DIR / "runtimes" / "subprocess" / self.name
        self.workdir.mkdir(parents=True)
        self.info.id = str(self.workdir)

    def _host_env(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}

    async def _zygote_for(
        self, argv: list[str], env: dict[str, str]
    ) -> zygote.Zygote | None:
        if not self.config.fork_server or not zygote.eligible(argv, sys.executable):
            return None
        server = await asyncio.to_thread(
            zygote.get_zygote, sys.executable, self.config.preload, self._host_env()
        )
        return server if server is not None and server.accepts(argv, env) else None

    async def _spawn(
        self,
        argv: list[str],
        env: dict[str, str],
        *,
        stdin: int | None,
        stdout: int | IO[bytes] | None,
        stderr: int | None,
    ) -> "asyncio.subprocess.Process | zygote.ZygoteProcess":
        """Start `argv` in its own session (so the whole tree can be signalled).

        Stdio arguments follow `asyncio.create_subprocess_exec`, restricted to
        PIPE, STDOUT (stderr only), an open file, or None to inherit."""
        full_env = self._host_env()
        full_env.update(self.process_env(env))
        server = await self._zygote_for(argv, full_env)
        # A fork needs a descriptor per stream (None asks it for a new pipe); a
        # stderr merged into a new stdout pipe is left to exec.
        merged_into_pipe = (
            stderr == asyncio.subprocess.STDOUT and stdout == asyncio.subprocess.PIPE
        )
        if server is not None and not merged_into_pipe:
            in_fd = _child_fd(stdin, 0)
            out_fd = _child_fd(stdout, 1)
            err_fd = (
                out_fd if stderr == asyncio.subprocess.STDOUT else _child_fd(stderr, 2)
            )
            try:
                return await server.spawn(
                    argv,
                    full_env,
                    str(self.workdir),
                    stdin=in_fd,
                    stdout=out_fd,
                    stderr=err_fd,
                )
            except (OSError, RuntimeError) as exc:
                self._fork_server_failed(exc)
        return await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.workdir,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

    _fork_warned: ClassVar[bool] = False

    @classmethod
    def _fork_server_failed(cls, exc: BaseException) -> None:
        if not cls._fork_warned:
            cls._fork_warned = True
            logger.warning("fork server spawn failed, using exec: %s", exc)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        proc = await self._spawn(
            argv,
            env,
            stdin=None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await proc.communicate()
        finally:
            # If the await didn't finish, the caller cancelled it (e.g. the rollout's
            # scoring_timeout / agent_timeout fired): communicate() leaves the process
            # running, so SIGKILL its whole group (start_new_session => pgid == pid) — otherwise
            # a hung child (a wedged uv/sympy verify) outlives the rollout and leaks CPU. A
            # no-op once it has exited on its own.
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return ProgramResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        proc = await self._spawn(
            argv,
            env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._background.append(proc)
        return SubprocessProcess(proc)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        logfile = self.workdir / log
        with logfile.open(
            "wb"
        ) as f:  # child dups the fd; safe to close ours after spawn
            proc = await self._spawn(
                argv, env, stdin=None, stdout=f, stderr=asyncio.subprocess.STDOUT
            )
        self._background.append(
            proc
        )  # killed in stop() — a host process won't die on its own

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is None:
            return await asyncio.to_thread((self.workdir / path).read_bytes)

        # Leave special files to the cancellable shell path without opening a
        # FIFO and waking its writer. A nonblocking open and descriptor check
        # also cover a regular path being replaced by a FIFO after the stat.
        def read() -> bytes | None:
            target = self.workdir / path
            if not stat.S_ISREG(target.stat().st_mode):
                return None
            fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as file:
                if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                    return None
                return file.read(max_bytes)

        try:
            data = await asyncio.to_thread(read)
        except OSError as exc:
            raise SandboxError(f"read {path!r}: {exc}") from exc
        return data if data is not None else await super()._read(path, max_bytes)

    async def write(self, path: str, data: bytes) -> None:
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def teardown(self) -> None:
        """Stop and reap background servers before their event loop closes."""
        background = list(self._background)
        for proc in background:
            signal_process(proc, signal.SIGTERM)
        if background:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(proc.wait() for proc in background), return_exceptions=True
                    ),
                    timeout=_BACKGROUND_STOP_TIMEOUT,
                )
            except TimeoutError:
                for proc in background:
                    signal_process(proc, signal.SIGKILL)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(
                            *(proc.wait() for proc in background),
                            return_exceptions=True,
                        ),
                        timeout=_BACKGROUND_STOP_TIMEOUT,
                    )
        self._background = []
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, True)

    def cleanup(self) -> None:
        for proc in self._background:
            signal_process(proc, signal.SIGTERM)
        self._background = []
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
