"""The numbers MODEL_EVAL.md quotes have to be the numbers the runs recorded.

The recorded runs live in MINI_BENCH*.json and the document summarises them, which
is exactly the arrangement that drifted in EXPERIMENTS.md. Writing this found four
errors in the document: a "median tokens" column that held totals, two tables
whose median-turns columns had been swapped by hand, and a median taken over an
even number of rows and rounded to a whole turn.

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


def test_the_document_says_how_many_tasks_the_suite_has(document):
    from bench import mini_bench

    assert f'{len(mini_bench.TASKS)} tasks' in document


def test_the_document_does_not_claim_a_lift_the_runs_did_not_show(document):
    """The conclusion is a negative result; the wording has to stay negative."""
    assert 'cannot rank configurations' in document or 'The ceiling did not move' in document
