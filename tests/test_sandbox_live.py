"""Live checks for run_sandbox, against a real Docker engine.

The rest of the sandbox tests assert the command line that would be built, which
is what keeps the suite runnable with no Docker. That leaves the claims the flags
are supposed to buy unverified: that the directory maps where it says, that
writes land on the host, that the root filesystem is read-only, that there is no
network, that memory is capped, that a container does not survive the call.

Those can only be checked against an engine, so this module skips itself unless
one is running and the image is present. It pulls nothing: the tool runs with
--pull=never, so an image that is missing is a skip, not a download.

    docker pull python:3.12-slim
    uv run pytest tests/test_sandbox_live.py
"""

import shutil
import subprocess
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_harness.tool import box  # noqa: E402

IMAGE = 'python:3.12-slim'
LABEL = 'mini-harness.sandbox=true'


def _docker() -> str|None:
    return shutil.which('docker')


def _engine_reason() -> str:
    """Why the live checks cannot run, or an empty string when they can."""
    docker = _docker()
    if not docker:
        return 'docker is not on PATH'
    try:
        done = subprocess.run([docker, 'version', '--format', '{{.Server.Version}}'],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f'docker did not answer: {type(error).__name__}'
    if done.returncode != 0:
        return 'no Docker engine is running'
    try:
        image = subprocess.run([docker, 'image', 'inspect', IMAGE],
                               capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f'docker did not answer: {type(error).__name__}'
    if image.returncode != 0:
        return f'{IMAGE} is not present locally, and the tool runs with --pull=never'
    return ''


REASON = _engine_reason()
pytestmark = pytest.mark.skipif(bool(REASON), reason=REASON or 'docker available')


def script(cfg, name: str, body: str) -> None:
    target = cfg.sandbox_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding='utf-8')


def containers_left() -> list:
    done = subprocess.run([_docker(), 'ps', '-a', '--filter', f'label={LABEL}',
                           '--format', '{{.Names}} {{.Status}}'],
                          capture_output=True, text=True, timeout=60)
    return [line for line in done.stdout.splitlines() if line.strip()]


# ------------------------------------------------------------------ it runs


def test_a_script_runs_and_its_output_comes_back(cfg):
    script(cfg, 'demo.py', 'import sys\nprint("hello from the sandbox", sys.version_info[:2])\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python demo.py'), cfg=cfg)

    assert 'hello from the sandbox' in output
    assert '(3, 12)' in output, 'the container should be the Python 3.12 image'


def test_the_container_is_a_different_machine(cfg):
    """The host's own interpreter path must not be the one that answered."""
    script(cfg, 'where.py', 'import sys\nprint(sys.executable)\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python where.py'), cfg=cfg)

    assert '/usr/local/bin/python' in output
    assert str(sys.executable) not in output


def test_stdout_and_stderr_are_both_returned(cfg):
    script(cfg, 'both.py', 'import sys\nprint("to stdout")\nprint("to stderr", file=sys.stderr)\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python both.py'), cfg=cfg)

    assert 'to stdout' in output
    assert 'to stderr' in output


# ------------------------------------------------------------------ the mount


def test_sandbox_is_mounted_at_workspace_and_writes_reach_the_host(cfg):
    script(cfg, 'write_it.py',
           'from pathlib import Path\n'
           'Path("made_in_container.txt").write_text("written by the container")\n'
           'print("wrote", Path("made_in_container.txt").resolve())\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python write_it.py'), cfg=cfg)

    assert '/workspace/made_in_container.txt' in output
    host_file = cfg.sandbox_dir / 'made_in_container.txt'
    assert host_file.read_text(encoding='utf-8') == 'written by the container'


def test_the_workspace_only_shows_the_sandbox_directory(cfg, workspace):
    """A file next to sandbox/ must not be visible from inside."""
    (workspace / 'outside.txt').write_text('not for the container', encoding='utf-8')
    script(cfg, 'list_it.py', 'from pathlib import Path\nprint(sorted(p.name for p in Path(".").iterdir()))\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python list_it.py'), cfg=cfg)

    assert 'outside.txt' not in output
    assert 'list_it.py' in output


# ------------------------------------------------------------------ the isolation flags


def test_the_root_filesystem_is_read_only(cfg):
    script(cfg, 'write_root.py',
           'try:\n'
           '    open("/etc/probe", "w").write("should not happen")\n'
           'except OSError as error:\n'
           '    print("refused:", error)\n'
           'else:\n'
           '    print("wrote to the root filesystem")\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python write_root.py'), cfg=cfg)

    assert 'refused:' in output
    assert 'wrote to the root filesystem' not in output


def test_tmp_is_writable_because_the_tool_mounts_it(cfg):
    """Home is pointed at /tmp, so a tool that needs scratch space still works."""
    script(cfg, 'write_tmp.py',
           'from pathlib import Path\n'
           'Path("/tmp/scratch.txt").write_text("ok")\n'
           'print("tmp is writable")\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python write_tmp.py'), cfg=cfg)

    assert 'tmp is writable' in output


def test_there_is_no_network(cfg):
    script(cfg, 'reach_out.py',
           'import socket\n'
           'try:\n'
           '    socket.create_connection(("1.1.1.1", 53), timeout=5)\n'
           'except OSError as error:\n'
           '    print("unreachable:", type(error).__name__)\n'
           'else:\n'
           '    print("network is reachable")\n')

    output = box.run_sandbox(box.RunSandboxInput(command='python reach_out.py',
                                                 timeout_seconds=20), cfg=cfg)

    assert 'unreachable:' in output
    assert 'network is reachable' not in output


def test_memory_is_capped(cfg):
    """--memory=256m must not allow a 512 MB allocation.

    Two outcomes are acceptable and both mean the cap held: Python raises
    MemoryError, or the kernel kills the container outright. The second is what
    actually happens here -- exit 137, "Killed" -- and the tool turns that into a
    RuntimeError carrying the container's output.
    """
    script(cfg, 'eat_memory.py',
           'try:\n'
           '    block = bytearray(512 * 1024 * 1024)\n'
           '    block[0] = 1\n'
           'except MemoryError:\n'
           '    print("refused: MemoryError")\n'
           'else:\n'
           '    print("allocated 512 MB")\n')

    try:
        output = box.run_sandbox(box.RunSandboxInput(command='python eat_memory.py',
                                                     timeout_seconds=60), cfg=cfg)
    except RuntimeError as error:
        output = str(error)

    assert 'allocated 512 MB' not in output
    assert 'MemoryError' in output or 'Killed' in output or '137' in output, output


# ------------------------------------------------------------------ failures and cleanup


def test_a_non_zero_exit_is_an_error(cfg):
    script(cfg, 'fail.py', 'import sys\nprint("about to fail")\nsys.exit(3)\n')

    with pytest.raises(RuntimeError, match='exited with code 3'):
        box.run_sandbox(box.RunSandboxInput(command='python fail.py'), cfg=cfg)


def test_a_command_that_overruns_its_timeout_is_stopped(cfg):
    script(cfg, 'slow.py', 'import time\nprint("starting")\ntime.sleep(60)\n')

    with pytest.raises(TimeoutError, match='exceeded 3s'):
        box.run_sandbox(box.RunSandboxInput(command='python slow.py', timeout_seconds=3), cfg=cfg)


def test_no_container_survives_the_call(cfg):
    script(cfg, 'demo.py', 'print("done")\n')

    box.run_sandbox(box.RunSandboxInput(command='python demo.py'), cfg=cfg)

    assert containers_left() == []


def test_no_container_survives_a_failure(cfg):
    script(cfg, 'fail.py', 'import sys\nsys.exit(4)\n')

    with pytest.raises(RuntimeError):
        box.run_sandbox(box.RunSandboxInput(command='python fail.py'), cfg=cfg)

    assert containers_left() == []
