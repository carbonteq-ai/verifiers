"""The subprocess runtime's fork server behaves like starting a new interpreter."""

import asyncio
import json
import shutil
import sys

import pytest

from verifiers.v1.runtimes import zygote
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="fork server needs POSIX fork"
)

PROBE = """
import json, os, sys, tempfile
print(json.dumps({
    "argv": sys.argv,
    "cwd": os.getcwd(),
    "path0": sys.path[0],
    "name": __name__,
    "marker": os.environ.get("MARKER"),
    "tmp": tempfile.gettempdir(),
    "preloaded": "colorsys" in sys.modules,
    "pgid_is_pid": os.getpgid(0) == os.getpid(),
}))
print("to-stderr", file=sys.stderr)
sys.exit(int(os.environ.get("EXIT", "0")))
"""


@pytest.fixture(autouse=True)
def _fresh_zygotes():
    yield
    zygote.close_all()


async def _runtime(fork_server: bool) -> SubprocessRuntime:
    # `tempfile` is preloaded so a cached zygote tempdir would leak into children.
    config = SubprocessConfig(fork_server=fork_server, preload=["colorsys", "tempfile"])
    runtime = SubprocessRuntime(config)
    await runtime.start()
    return runtime


async def _probe(runtime: SubprocessRuntime, argv_tail: list[str], env=None):
    result = await runtime.run([sys.executable, *argv_tail], env or {})
    return result, json.loads(result.stdout.splitlines()[0])


@pytest.mark.parametrize("fork_server", [False, True])
async def test_program_sees_a_fresh_interpreter(fork_server, tmp_path):
    runtime = await _runtime(fork_server)
    try:
        script = tmp_path / "probe.py"
        script.write_text(PROBE)
        env = {"MARKER": "m", "EXIT": "3", "TMPDIR": str(runtime.workdir)}

        result, seen = await _probe(runtime, [str(script), "a", "b"], env)
        assert result.exit_code == 3
        assert result.stderr.strip() == "to-stderr"
        assert seen["argv"] == [str(script), "a", "b"]
        assert seen["cwd"] == str(runtime.workdir)
        assert seen["path0"] == str(tmp_path)
        assert seen["name"] == "__main__"
        assert seen["marker"] == "m"
        assert seen["tmp"] == str(runtime.workdir)
        assert seen["pgid_is_pid"]
        assert seen["preloaded"] is fork_server

        result, seen = await _probe(runtime, ["-c", PROBE, "x"], {"EXIT": "0"})
        assert result.exit_code == 0
        assert seen["argv"] == ["-c", "x"]
        assert seen["path0"] == ""
        assert seen["name"] == "__main__"
    finally:
        await runtime.teardown()


@pytest.mark.parametrize("fork_server", [False, True])
async def test_module_program_and_background_log(fork_server, tmp_path):
    runtime = await _runtime(fork_server)
    try:
        package = runtime.workdir / "probe_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "tool.py").write_text(PROBE)
        result, seen = await _probe(runtime, ["-m", "probe_pkg.tool", "z"])
        assert result.exit_code == 0
        assert seen["argv"] == [str(package / "tool.py"), "z"]
        assert seen["path0"] == str(runtime.workdir)

        await runtime.run_background(
            [sys.executable, "-m", "probe_pkg.tool"], {"MARKER": "bg"}, "bg.log"
        )
        await asyncio.wait_for(runtime._background[-1].wait(), 30)
        lines = (runtime.workdir / "bg.log").read_text().splitlines()
        # stderr is merged into the log (and line-buffered, so it lands first).
        assert sorted(lines, key=len) == ["to-stderr", lines[-1]]
        assert json.loads(lines[-1])["marker"] == "bg"
    finally:
        await runtime.teardown()


@pytest.mark.parametrize("fork_server", [False, True])
async def test_interactive_process_pipes_and_signals(fork_server):
    runtime = await _runtime(fork_server)
    try:
        echo = "import sys\nfor line in sys.stdin:\n    print(line.upper(), end='', flush=True)\n"
        process = await runtime.open_process([sys.executable, "-c", echo], {})
        await process.write(b"hello\n")
        chunk = await asyncio.wait_for(anext(process.stdout), 30)
        assert chunk == b"HELLO\n"

        sleeper = await runtime.open_process(
            [sys.executable, "-c", "import time; time.sleep(60)"], {}
        )
        await sleeper.terminate()
        assert await asyncio.wait_for(sleeper.wait(), 30) == -15
        await process.kill()
        assert await asyncio.wait_for(process.wait(), 30) == -9
    finally:
        await runtime.teardown()


async def test_ineligible_command_lines_exec(tmp_path):
    runtime = await _runtime(True)
    try:
        # Interpreter options and other executables are not forked.
        result, seen = await _probe(runtime, ["-u", "-c", PROBE])
        assert result.exit_code == 0
        assert seen["preloaded"] is False
        result = await runtime.run(["sh", "-c", "echo $MARKER"], {"MARKER": "sh"})
        assert result.stdout.strip() == "sh"
        # A startup variable the zygote was not started with forces exec.
        result, seen = await _probe(runtime, ["-c", PROBE], {"PYTHONHASHSEED": "7"})
        assert seen["preloaded"] is False
    finally:
        await runtime.teardown()


async def test_many_concurrent_programs():
    runtime = await _runtime(True)
    try:
        results = await asyncio.gather(
            *(runtime.run([sys.executable, "-c", f"print({i})"], {}) for i in range(64))
        )
        assert [r.stdout.strip() for r in results] == [str(i) for i in range(64)]
        assert all(r.exit_code == 0 for r in results)
    finally:
        await runtime.teardown()


def test_same_interpreter_requires_the_same_venv(tmp_path):
    base = tmp_path / "base" / "python3.12"
    base.parent.mkdir()
    base.write_text("")
    for venv in ("a", "b"):
        (tmp_path / venv / "bin").mkdir(parents=True)
        for name in ("python", "python3"):
            (tmp_path / venv / "bin" / name).symlink_to(base)
    a, a3, b = (
        str(tmp_path / p) for p in ("a/bin/python", "a/bin/python3", "b/bin/python")
    )
    assert zygote.same_interpreter(a3, a)
    assert not zygote.same_interpreter(b, a)  # same base binary, other venv
    assert zygote.eligible([a3, "-m", "x"], a)
    assert not zygote.eligible([a, "-u", "x.py"], a)


UV_PROBE = """# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import json, os, sys
print(json.dumps({
    "executable": sys.executable,
    "venv": os.environ.get("VIRTUAL_ENV"),
    "path_head": os.environ["PATH"].split(os.pathsep)[0],
    "depth": os.environ.get("UV_RUN_RECURSION_DEPTH"),
    "argv": sys.argv[1:],
    "preloaded": "colorsys" in sys.modules,
}))
"""


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
@pytest.mark.parametrize("fork_server", [False, True])
async def test_uv_script_programs_fork_from_their_environment(fork_server):
    runtime = await _runtime(fork_server)
    try:
        argv = await runtime.prepare_uv_script(UV_PROBE)
        result = await runtime.run([*argv, "x"], {})
        assert result.exit_code == 0, result.stderr
        seen = json.loads(result.stdout.splitlines()[0])
        venv = argv[4]
        assert seen["venv"] == venv
        assert seen["path_head"] == f"{venv}/bin"
        assert seen["depth"] == "1"
        assert seen["argv"] == ["x"]
        assert seen["executable"].startswith(venv)
        assert seen["preloaded"] is fork_server
    finally:
        await runtime.teardown()


@pytest.mark.parametrize("fork_server", [False, True])
async def test_module_programs_are_learned_by_the_zygote(
    fork_server, tmp_path, monkeypatch
):
    package = tmp_path / "learned_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "tool.py").write_text(
        "import json, sys\n"
        "print(json.dumps({'warm': 'learned_pkg.tool' in sys.modules, 'name': __name__}))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    runtime = await _runtime(fork_server)
    try:
        for _ in range(2):
            result = await runtime.run([sys.executable, "-m", "learned_pkg.tool"], {})
            assert result.exit_code == 0, result.stderr
            seen = json.loads(result.stdout)
            # Imported once in the zygote, then run fresh as __main__ per fork.
            assert seen == {"warm": fork_server, "name": "__main__"}
    finally:
        await runtime.teardown()
