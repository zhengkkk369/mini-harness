"""The test suite, as a package.

This file exists because of a real CI failure: without it, pytest's default
import mode inserts `tests/` on `sys.path` rather than the repository root, so
`from tests.conftest import ...` only worked when something else happened to put
the root there first -- which `python -m pytest` does for you and the `pytest`
entry point the CI uses does not. Collection therefore failed on the first two
modules alphabetically, and only in CI.

Making `tests` a package makes the root the inserted path, so the import works
the same way however pytest is started.
"""
