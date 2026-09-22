"""The numbers MODEL_EVAL.md quotes have to be the numbers the runs recorded.

The recorded runs live in MINI_BENCH*.json and the document summarises them, which
is exactly the arrangement that drifted in EXPERIMENTS.md. Writing this found four
errors in the document: a "median tokens" column that held totals, two tables
whose median-turns columns had been swapped by hand, and a median taken over an
even number of rows and rounded to a whole turn. It later found a fifth, of the
other shape: the table that counts every recorded run still described an earlier
repository (45/8/69 against 146), because no per-configuration check could see a
claim that spans the artifacts.

`bench/mini_bench.aggregate` owns the definition of a summary, so the checker
cannot disagree with the runner about what a column means.
"""

import json
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.mini_bench import aggregate, summary_row  # noqa: E402

DOC = ROOT / 'MODEL_EVAL.md'

# The artifacts whose per-configuration summaries the document carries as rows in
# the canonical shape.
TABLE_ARTIFACTS = ('MINI_BENCH_HARD.json', 'MINI_BENCH_NUDGE.json',
                   'MINI_BENCH_LONG.json', 'MINI_BENCH_WEAK.json')
# The older two predate that shape: the unpriced 72-run aggregate has no cost to
# show and the priced one is a single configuration with its own columns. Their
# values are still checked, just not as a whole row.
VALUE_ARTIFACTS = ('MINI_BENCH.json', 'MINI_BENCH_PRICED.json')

pytestmark = pytest.mark.skipif(not DOC.exists(), reason='the document must be present')


def read(name: str) -> list:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f'{name} is not recorded here')
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def document():
    return DOC.read_text(encoding='utf-8')


def configs_of(rows: list) -> list:
    seen = []
    for row in rows:
        if row['config'] not in seen:
            seen.append(row['config'])
    return seen


@pytest.mark.parametrize('artifact', TABLE_ARTIFACTS)
def test_every_recorded_summary_row_is_in_the_document(artifact, document):
    rows = read(artifact)
    missing = [summary_row(aggregate(rows, config)) for config in configs_of(rows)
               if summary_row(aggregate(rows, config)) not in document]

    assert not missing, f'{artifact}: MODEL_EVAL.md does not carry {missing}'


@pytest.mark.parametrize('artifact', VALUE_ARTIFACTS)
def test_the_older_tables_quote_the_recorded_values(artifact, document):
    rows = read(artifact)
    for config in configs_of(rows):
        summary = aggregate(rows, config)
        assert summary['passed'] in document, f'{artifact}/{config}: {summary["passed"]}'
        assert f'{summary["total_tokens"]:,}' in document or summary['total_cost'] == 0, (
            f'{artifact}/{config}: total tokens {summary["total_tokens"]:,} appear nowhere')
        if summary['total_cost']:
            assert f'${summary["total_cost"]:.4f}' in document, f'{artifact}/{config}: cost'
        # The priced artifact is summarised by a hand-written table rather than by
        # a canonical row, and that is where a stale median turns hid: every other
        # column of this run was checked, so 11 sat there against a recorded 10.5.
        if artifact == 'MINI_BENCH_PRICED.json':
            assert f"{summary['median_turns']:g}" in document, (
                f'{artifact}/{config}: median turns {summary["median_turns"]:g} appear nowhere')


def test_the_recorded_retrieval_measurement_is_in_the_document(document):
    """A provider-backed measurement has to be documented, not just recorded.

    The artifact is committed, so a run that improved or broke retrieval quality
    shows up here as a document that no longer matches it.
    """
    from bench.embed_quality import QUALITY_HEADER, quality_rows

    path = ROOT / 'EMBED_QUALITY.json'
    if not path.exists():
        pytest.skip('no retrieval measurement recorded here')
    artifact = json.loads(path.read_text(encoding='utf-8'))
    if artifact.get('offline') or not artifact.get('scenarios'):
        pytest.skip('only a provider run says anything about an embedding model')

    assert QUALITY_HEADER in document, 'MODEL_EVAL.md does not carry the retrieval table header'
    assert artifact['model'] in document, f"{artifact['model']} is not named in the document"
    missing = [row for row in quality_rows(artifact) if row not in document]

    assert not missing, 'MODEL_EVAL.md does not carry:\n' + '\n'.join(missing)


def test_the_document_says_how_many_tasks_the_suite_has(document):
    from bench import mini_bench

    assert f'{len(mini_bench.TASKS)} tasks' in document


def test_the_census_table_counts_every_recorded_run(document):
    """The cross-artifact table is the one no per-configuration check covers.

    It claims to count every run in the repository, and it was wrong: two more
    tiers were recorded and it still read 45/8/69 against 146. Deriving it from
    every artifact is what keeps the claim true.
    """
    from bench.mini_bench import census_rows

    paths = sorted(ROOT.glob('MINI_BENCH*.json'))
    assert paths, 'no recorded runs to count'
    rows = census_rows(paths)
    missing = [row for row in rows if row not in document]

    assert not missing, 'MODEL_EVAL.md does not carry:\n' + '\n'.join(missing)
    counted = sum(int(row.split('|')[1]) for row in rows)
    recorded = sum(len(read(path.name)) for path in paths)
    assert counted == recorded, f'the census counts {counted} of {recorded} runs'


def test_the_document_does_not_claim_a_lift_the_runs_did_not_show(document):
    """The conclusion is a negative result; the wording has to stay negative."""
    assert 'cannot rank configurations' in document or 'The ceiling did not move' in document
