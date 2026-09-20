"""Path guards: reads stay in the workspace, writes stay in sandbox/."""

import pytest

from mini_harness.tool.path import _resolve_file, is_denied, validate_read, validate_write
from tests.conftest import write


def test_relative_read_resolves_against_workspace(cfg, workspace):
    write(workspace / "pkg" / "mod.py", "x = 1\n")
    assert validate_read("pkg/mod.py", cfg=cfg) == (workspace / "pkg" / "mod.py").resolve()


def test_read_outside_workspace_is_refused(cfg, tmp_path):
    outside = write(tmp_path / "outside.txt", "secret\n")
    with pytest.raises(PermissionError, match="Access denied"):
        validate_read(outside, cfg=cfg)


def test_read_escape_via_parent_traversal_is_refused(cfg, tmp_path):
    write(tmp_path / "outside.txt", "secret\n")
    with pytest.raises(PermissionError, match="Access denied"):
        validate_read("../outside.txt", cfg=cfg)


@pytest.mark.parametrize(
    "name",
    [".env", ".env.local", "server.pem", "id_rsa", "id_ed25519", ".netrc", "aws_secret.txt"],
)
def test_sensitive_file_names_are_denied(cfg, workspace, name):
    target = write(workspace / name, "credential\n")
    assert is_denied(target, cfg=cfg)
    with pytest.raises(PermissionError, match="denied"):
        validate_read(name, cfg=cfg)


def test_sensitive_directories_are_denied(cfg, workspace):
    target = write(workspace / ".ssh" / "config", "Host *\n")
    assert is_denied(target, cfg=cfg)


def test_denied_dir_check_does_not_match_a_file_itself(cfg, workspace):
    """A file literally named .aws inside the workspace is fine to read."""
    target = write(workspace / ".aws", "not a directory\n")
    assert not is_denied(target, cfg=cfg)


def test_guard_read_off_allows_outside_reads(cfg_factory, tmp_path):
    outside = write(tmp_path / "outside.txt", "ok\n")
    assert validate_read(outside, cfg=cfg_factory(guard_read=False)) == outside.resolve()


def test_write_inside_sandbox_is_allowed(cfg, workspace):
    assert validate_write("sandbox/out.py", cfg=cfg) == (workspace / "sandbox" / "out.py").resolve()


def test_write_outside_sandbox_is_refused(cfg, workspace):
    for target in ["evil.py", str(workspace / "src" / "evil.py")]:
        with pytest.raises(PermissionError, match="Access denied"):
            validate_write(target, cfg=cfg)


def test_write_escape_via_parent_traversal_is_refused(cfg, cfg_factory):
    with pytest.raises(PermissionError, match="Access denied"):
        validate_write("sandbox/../evil.py", cfg=cfg_factory(guard_write=True))


def test_guard_write_off_allows_any_path(cfg_factory, tmp_path):
    loose = cfg_factory(guard_write=False)
    assert validate_write(tmp_path / "anywhere" / "f.py", cfg=loose) == (
        tmp_path / "anywhere" / "f.py"
    ).resolve()


def test_resolve_file_keeps_absolute_paths(cfg, workspace):
    absolute = workspace / "sandbox" / "a.py"
    assert _resolve_file(absolute, cfg=cfg) == absolute.resolve()
