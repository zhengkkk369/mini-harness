"""The recorded numbers in EXPERIMENTS.md have to be the recorded run.

EXPERIMENTS.md is hand-written prose around tables, and EXPERIMENTS.json is the
artifact `bench/experiments.py` writes. They drifted once: the experiment was
re-run, the JSON was overwritten, and only one of the tables was updated, so the
document quoted timings from a run that no longer existed. The document is what a
reader trusts, so each table row is rebuilt from the JSON and required to appear
in the prose.

`bench/sync_experiments_doc.py` holds the row formats and can rewrite them, so
checking and fixing cannot disagree about what a row should look like.

The same thing then happened to the *paragraphs*: the rows were re-recorded and
the sentences next to them kept quoting the previous run. So the sentences that
quote a timed number are generated from the artifact as well, and the test below
requires them verbatim.

The live summariser is the one experiment that reaches the network, so this file
also holds the guard on it: the runner's placeholder key must never be mistaken
for a real one.
"""

import json
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.sync_experiments_doc import expected_rows, missing_prose, render_prose, render_rows  # noqa: E402

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


def test_every_recorded_section_can_be_recorded_again(recorded):
    """A section the runner cannot re-record is a section that will drift.

    `--only` selects from a fixed list, so an experiment added to the artifact but
    not to that list could never be re-measured on its own: the next re-record
    would either move every other number or leave this one stale.
    """
    from bench.experiments import SECTIONS

    metadata = {'python', 'platform', 'repeats'}

    assert set(recorded) == set(SECTIONS) | metadata, (
        'the artifact and the runner disagree about which experiments exist')


def test_sync_adds_a_row_the_document_is_missing(tmp_path):
    """The fixer has to be able to add, not only rewrite.

    Re-recording compaction added a summariser: the artifact grew a row per size,
    the document had no line for it, and the old sync wrote the artifact's first
    rows over the document's and dropped the rest -- losing a whole group and
    leaving the checker to notice.
    """
    import json as json_module

    from bench.sync_experiments_doc import sync

    artifact = json_module.loads(Path(ARTIFACT).read_text(encoding='utf-8'))
    document = DOC.read_text(encoding='utf-8')
    # Drop one recorded row and the line that carries it, then sync.
    row = render_rows(artifact)['compaction'][0]
    doctored = tmp_path / 'EXPERIMENTS.md'
    doctored.write_text('\n'.join(line for line in document.splitlines() if line != row) + '\n',
                        encoding='utf-8')

    changed, unplaced = sync(doc=doctored, artifact=ARTIFACT)

    assert changed >= 1
    assert row in doctored.read_text(encoding='utf-8'), 'the missing row was not added back'
    assert not [item for item in unplaced if item[0] == 'compaction']



def test_the_document_carries_every_generated_sentence(recorded, document):
    """Prose that quotes a timed number has to be the current recording.

    This is the half that drifted: the table was updated and the sentence above
    it was not, so the document contradicted itself in adjacent lines.
    """
    missing = missing_prose(recorded, document)

    assert not missing, ('EXPERIMENTS.md does not carry these sentences from the artifact:\n'
                         + '\n'.join(missing))


def test_the_document_does_not_quote_the_rows_it_generates():
    """A restated number is the drift path; the source is read instead."""
    from bench.sync_experiments_doc import _overheads

    recorded = json.loads(ARTIFACT.read_text(encoding='utf-8'))
    document = DOC.read_text(encoding='utf-8')
    rows = _overheads(recorded)
    stale = []
    for row in rows.values():
        for number in (f"{row['delta_ms']:+.2f} ms", f"{row['min_delta_ms']:+.2f} ms",
                       f"{row['max_delta_ms']:+.2f} ms"):
            for line in document.splitlines():
                if line.strip().startswith('| ') or number not in line:
                    continue
                stale.append((number, line.strip()))
    generated = {line.strip() for line in render_prose(recorded)}
    stale = [(number, line) for number, line in stale if line not in generated]

    assert not stale, 'these sentences restate a generated timing:\n' + '\n'.join(
        f'{number}: {line}' for number, line in stale)


def test_sync_rewrites_a_sentence_that_quotes_an_old_run(tmp_path):
    """The fixer has to repair prose, not only rows."""
    from bench.sync_experiments_doc import sync

    recorded = json.loads(ARTIFACT.read_text(encoding='utf-8'))
    document = DOC.read_text(encoding='utf-8')
    stale = render_prose(recorded)[1].replace('+74.62 ms', '+999.99 ms')
    doctored = tmp_path / 'EXPERIMENTS.md'
    doctored.write_text(document.replace(render_prose(recorded)[1], stale), encoding='utf-8')

    changed, unplaced = sync(doc=doctored, artifact=ARTIFACT)

    assert render_prose(recorded)[1] in doctored.read_text(encoding='utf-8')
    assert changed >= 1
    assert not [item for item in unplaced if item[0] == 'prose']


# --------------------------------------------------------------- the live summariser


def test_the_runner_placeholder_is_not_a_credential(monkeypatch, tmp_path):
    """`experiments.py` must not mistake its own offline placeholder for a key.

    It sets `DEEPSEEK_API_KEY=experiments-offline` so that no offline experiment
    can reach the network by accident. The live summariser is the one thing that
    should reach it, and only with a key that came from somewhere real.
    """
    from bench.experiments import _real_key

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'experiments-offline')

    assert _real_key(tmp_path / 'absent.env') == ''


def test_the_live_summariser_refuses_to_start_without_a_key(monkeypatch, tmp_path):
    from bench.experiments import LiveSummary

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'experiments-offline')

    with pytest.raises(RuntimeError, match='DEEPSEEK_API_KEY'):
        LiveSummary(env_path = tmp_path / 'absent.env')


def test_a_key_in_a_local_env_file_is_used(monkeypatch, tmp_path):
    from bench.experiments import _real_key

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'experiments-offline')
    env = tmp_path / '.env'
    env.write_text('SOMETHING_ELSE=1\nDEEPSEEK_API_KEY="sk-from-file"\n', encoding='utf-8')

    assert _real_key(env) == 'sk-from-file'


def test_a_real_environment_key_wins_and_needs_no_file(monkeypatch, tmp_path):
    from bench.experiments import _real_key

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-from-environment')

    assert _real_key(tmp_path / 'absent.env') == 'sk-from-environment'


