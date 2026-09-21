"""The recorded numbers in EXPERIMENTS.md have to be the recorded run.

EXPERIMENTS.md is hand-written prose around tables, and EXPERIMENTS.json is the
artifact `bench/experiments.py` writes. They drifted once: the experiment was
re-run, the JSON was overwritten, and only one of the tables was updated, so the
document quoted timings from a run that no longer existed. The document is what a
reader trusts, so each table row is rebuilt from the JSON and required to appear
in the prose.

`bench/sync_experiments_doc.py` holds the row formats and can rewrite them, so
checking and fixing cannot disagree about what a row should look like.
"""

import json
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.sync_experiments_doc import expected_rows, render_rows  # noqa: E402

ARTIFACT = ROOT / 'EXPERIMENTS.json'
DOC = ROOT / 'EXPERIMENTS.md'

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists() or not DOC.exists(),
    reason='the recorded run and the document it describes must both be present')


@pytest.fixture(scope='module')
def recorded():
    return json.loads(ARTIFACT.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def document():
    return DOC.read_text(encoding='utf-8')


def test_the_document_carries_every_recorded_row(recorded, document):
    missing = [row for row in expected_rows(recorded) if row not in document]

    assert not missing, 'EXPERIMENTS.md does not carry these recorded rows:\n' + '\n'.join(missing)


def test_the_document_reports_the_recorded_repeat_count(recorded, document):
    assert f"| Repeats | {recorded['repeats']} per timed configuration" in document


def test_the_document_reports_the_recorded_python_and_platform(recorded, document):
    assert f"| Python | {recorded['python']} |" in document
    assert recorded['platform'] in document or 'see `platform` in' in document


def test_the_structured_sections_are_present(recorded, document):
    """A section that stops being rendered should not pass unnoticed."""
    for key in ('parallel', 'subagents', 'trace', 'verify', 'recall', 'compaction',
                'compaction_repeat', 'vector', 'policy', 'exposure', 'recovery'):
        assert recorded[key], f'{key} recorded nothing'


def test_every_recorded_table_has_a_header_in_the_document(recorded, document):
    """A table nobody can place would be silently unchecked."""
    from bench.sync_experiments_doc import HEADERS, _key

    styled = _key(document)
    for table, rows in render_rows(recorded).items():
        if rows:
            assert any(_key(spelling) in styled for spelling in HEADERS[table]), table

