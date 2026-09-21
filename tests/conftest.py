"""Shared offline fixtures for the mini-harness test suite.

Nothing here touches the network: the API key is a dummy and every test that
would call the model injects a fake client instead.

The ``workspace`` and ``session_dir`` fixtures deliberately do not use pytest's
``tmp_path``. Under some file-access policies (including the sandbox this suite
was developed in) directories created by ``tempfile.mkdtemp`` are read-only:
files cannot be created inside them. These fixtures instead build plain
directories and remove them afterwards.
"""

import os
import shutil
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy-key")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from mini_harness.config import Config  # noqa: E402
from mini_harness.budget import ACCOUNT  # noqa: E402
from mini_harness.selector import SELECTION  # noqa: E402
from mini_harness.tool import box  # noqa: E402
from mini_harness.tool.block import TODO  # noqa: E402
from mini_harness.trace import TRACE  # noqa: E402

REGISTRY = {tool.name: tool for tool in box.TOOLS}
ALWAYS_ALLOW = box._always_allow
ALWAYS_DENY = box._for_sub
SCRATCH = ROOT / ".pytest-work"


@pytest.fixture(autouse=True)
def isolated_trace():
    """The trace is a module-level singleton; never leak a path between tests."""
    TRACE.configure(None)
    yield
    TRACE.configure(None)


@pytest.fixture(scope="session", autouse=True)
def scratch_root():
    """Session scratch area, removed on the way out."""
    # pytest's own basetemp (if a previous run created one) can be unreadable.
    shutil.rmtree(ROOT / ".pytest-tmp", ignore_errors=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    yield SCRATCH
    shutil.rmtree(SCRATCH, ignore_errors=True)


@pytest.fixture
def tmp_path(scratch_root):
    """Replacement for pytest's tmp_path.

    Same contract, but the directory is created by ``mkdir`` rather than
    ``mkdtemp`` so it stays writable under a restrictive file policy.
    """
    path = scratch_root / f"{uuid.uuid4().hex[:12]}-work"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_todo():
    """run_todo keeps module-level state; do not leak it between tests."""
    TODO.items = []
    yield
    TODO.items = []


@pytest.fixture(autouse=True)
def clean_selection():
    """The tool catalogue and the find_tools pins are process-wide too."""
    SELECTION.reset()
    yield
    SELECTION.reset()


@pytest.fixture(autouse=True)
def clean_accounting():
    """The run ledger is process-wide; a leaked budget would bill the next test."""
    ACCOUNT.reset()
    yield
    ACCOUNT.reset()


@pytest.fixture
def session_dir(tmp_path):
    """Where the agent under test writes session.json and its audit log."""
    path = tmp_path / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def cfg_factory(tmp_path, session_dir):
    """Build a Config rooted at a scratch workspace.

    ``tmp_path`` is never /logs, which also covers the bench profile's
    session-path fallback.
    """
    workspace = tmp_path / "ws"
    (workspace / "sandbox").mkdir(parents=True)

    def build(**overrides):
        base = Config(
            work_space=workspace,
            session_path=str(session_dir / "session.json"),
            guard_read=True,
            guard_write=True,
        )
        return replace(base, **overrides) if overrides else base

    build.workspace = workspace
    return build


@pytest.fixture
def workspace(cfg_factory):
    return cfg_factory.workspace


@pytest.fixture
def cfg(cfg_factory):
    return cfg_factory()


@pytest.fixture
def patch_openai(monkeypatch):
    """Make DeepSeekAgent.run_task construct our fake client.

    run_task builds its own OpenAI client internally, so the module-level
    symbol is the seam. Returns the MonkeyPatch, which is usable as a context
    manager that restores the original constructor on exit.
    """
    def patch(client):
        monkeypatch.setattr("mini_harness.agent.OpenAI", lambda **_: client)
        return monkeypatch

    return patch


def call(name, call_id=None, **arguments):
    """A stand-in for an OpenAI tool_call object.

    Real call ids are unique; pass ``call_id`` when a test issues the same tool
    more than once and something keys on the id.
    """
    import json

    return SimpleNamespace(
        id=call_id or f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def executor(cfg, confirm=ALWAYS_ALLOW):
    return box.ToolExecution(REGISTRY, confirm, cfg=cfg)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
