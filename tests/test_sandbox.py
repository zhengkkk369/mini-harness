"""run_sandbox: container isolation flags and cross-platform command building.

These tests fake Docker itself, so they need neither a daemon nor the image.
"""

import io
from types import SimpleNamespace

import pytest

from mini_harness.tool import box
from mini_harness.tool.box import RunSandboxInput


class FakeProcess:
    def __init__(self, command, payload=b"", returncode=0, **kwargs):
        self.args = command
        self.stdout = io.BytesIO(payload)
        self.stderr = b""
        self.returncode = returncode
        self.kwargs = kwargs
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True

    @property
    def pid(self):
        return 4242


@pytest.fixture
def docker(monkeypatch):
    """Pretend docker exists and capture every container invocation."""
    calls = {"popen": [], "run": []}

    def fake_popen(command, **kwargs):
        calls["popen"].append((command, kwargs))
        process = FakeProcess(command, b"sandbox output\n", **kwargs)
        return process

    def fake_run(command, **kwargs):
        calls["run"].append((command, kwargs))
        return box.subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(box.shutil, "which", lambda name: r"/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(box.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(box.subprocess, "run", fake_run)
    return calls


def test_missing_docker_is_explained(cfg, monkeypatch):
    monkeypatch.setattr(box.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires Docker"):
        box.run_sandbox(RunSandboxInput(command="python -c 'print(1)'"), cfg=cfg)


def test_container_is_isolated(cfg, docker):
    box.run_sandbox(RunSandboxInput(command="python -c 'print(1)'"), cfg=cfg)
    command = " ".join(docker["popen"][0][0])

    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--pids-limit=64" in command
    assert "--memory=256m" in command
    assert "--cpus=1" in command
    assert "--pull=never" in command


def test_only_the_sandbox_directory_is_mounted(cfg, workspace, docker):
    box.run_sandbox(RunSandboxInput(command="ls"), cfg=cfg)
    command = " ".join(docker["popen"][0][0])

    assert f"src={(workspace / 'sandbox').resolve()},dst=/workspace" in command
    assert str(workspace) + ",dst" not in command


def test_host_environment_is_not_forwarded(cfg, monkeypatch, docker):
    monkeypatch.setenv("MINI_HARNESS_FAKE_SECRET", "leak")
    box.run_sandbox(RunSandboxInput(command="env"), cfg=cfg)

    env_flags = [a for a in docker["popen"][0][0] if a.startswith("MINI_HARNESS_FAKE")]
    assert env_flags == []
    assert docker["popen"][0][1]["env"].get("MINI_HARNESS_FAKE_SECRET") is None


def test_output_is_returned(cfg, docker):
    assert box.run_sandbox(RunSandboxInput(command="echo hi"), cfg=cfg).strip() == "sandbox output"


def test_container_is_force_removed_afterwards(cfg, docker):
    box.run_sandbox(RunSandboxInput(command="echo hi"), cfg=cfg)

    cleanup = docker["run"][0][0]
    assert cleanup[:3] == [r"/usr/bin/docker", "rm", "-f"]
    assert any(part.startswith("mini-harness-") for part in cleanup)


def test_startup_timeout_is_reported(cfg, docker, monkeypatch):
    calls = {"n": 0}

    def hanging_process(command, **kwargs):
        def wait(timeout=None):
            calls["n"] += 1
            if calls["n"] == 1 and timeout is not None:
                raise box.subprocess.TimeoutExpired(command, timeout)
            return 0

        return SimpleNamespace(stdout=io.BytesIO(b""), wait=wait, poll=lambda: None, kill=lambda: None)

    monkeypatch.setattr(box.subprocess, "Popen", hanging_process)

    with pytest.raises(TimeoutError, match="startup or execution timed out"):
        box.run_sandbox(RunSandboxInput(command="sleep 99"), cfg=cfg)

    # the container is still reclaimed even though startup timed out
    assert docker["run"][0][0][:3] == [r"/usr/bin/docker", "rm", "-f"]


def test_inner_timeout_exit_code_is_reported(cfg, docker, monkeypatch):
    monkeypatch.setattr(box.subprocess, "Popen",
                        lambda command, **kwargs: FakeProcess(command, returncode=124, **kwargs))

    with pytest.raises(TimeoutError, match="exceeded 5s"):
        box.run_sandbox(RunSandboxInput(command="sleep 99", timeout_seconds=5), cfg=cfg)


def test_nonzero_exit_code_is_reported(cfg, docker, monkeypatch):
    monkeypatch.setattr(box.subprocess, "Popen",
                        lambda command, **kwargs: FakeProcess(command, b"boom\n", returncode=2, **kwargs))

    with pytest.raises(RuntimeError, match="exited with code 2"):
        box.run_sandbox(RunSandboxInput(command="false"), cfg=cfg)


def test_a_symlinked_sandbox_directory_is_refused(cfg, workspace):
    import shutil as real_shutil

    real_shutil.rmtree(workspace / "sandbox")
    (workspace / "real-sandbox").mkdir()
    try:
        (workspace / "sandbox").symlink_to(workspace / "real-sandbox")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this platform")

    with pytest.raises(PermissionError, match="must be a real directory"):
        box.run_sandbox(RunSandboxInput(command="ls"), cfg=cfg)


def test_a_sandbox_path_containing_a_comma_is_refused(cfg, tmp_path):
    from dataclasses import replace

    comma_cfg = replace(cfg, work_space=tmp_path / "with,comma")
    (comma_cfg.work_space / "sandbox").mkdir(parents=True)

    with pytest.raises(ValueError, match="cannot contain commas"):
        box.run_sandbox(RunSandboxInput(command="ls"), cfg=comma_cfg)


def test_timeout_seconds_is_bounded(cfg):
    from pydantic import ValidationError

    assert RunSandboxInput(command="ls").timeout_seconds == 30
    with pytest.raises(ValidationError):
        RunSandboxInput(command="ls", timeout_seconds=0)
    with pytest.raises(ValidationError):
        RunSandboxInput(command="ls", timeout_seconds=301)
