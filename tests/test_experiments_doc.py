"""The recorded numbers in EXPERIMENTS.md have to be the recorded run.

EXPERIMENTS.md is hand-written prose around tables, and EXPERIMENTS.json is the
artifact `bench/experiments.py` writes. They drifted once: the experiment was
re-run, the JSON was overwritten, and only one of the tables was updated, so the
document quoted timings from a run that no longer existed. The document is what a
reader trusts, so each table row is now rebuilt from the JSON and required to
appear in the prose.

This checks the numbers, not the sentences: prose can be edited freely, a value
cannot drift silently.
"""

import json

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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


def expected_rows(results):
    """Every numeric table row the document is required to carry."""
    rows = []
    for row in results['parallel']:
        rows.append(f"| {row['latency_ms']:.0f} ms | {row['calls']} | "
                    f"{row['serial']['median_ms']:.2f} ms | {row['parallel']['median_ms']:.2f} ms | "
                    f"{row['speedup']:.2f}x |")
    for row in results['subagents']:
        rows.append(f"| {row['latency_ms']:.0f} ms | {row['calls']} | "
                    f"{row['serial']['median_ms']:.2f} ms | {row['parallel']['median_ms']:.2f} ms | "
                    f"{row['speedup']:.2f}x |")
    for row in results['trace']:
        if 'median_ms' in row:
            rows.append(f"| {row['trace']} | {row['turns']} | {row['events']} | "
                        f"{row['median_ms']:.2f} ms | {row['min_ms']:.2f} ms | "
                        f"{row['max_ms']:.2f} ms | {row['bytes']:,} |")
        else:
            rows.append(f"| {row['turns']} | {row['events']} | **{row['delta_ms']:+.2f} ms** | "
                        f"{row['min_delta_ms']:+.2f} ms | {row['max_delta_ms']:+.2f} ms | "
                        f"{row['per_event_us']:.1f} us |")
    for row in results['verify']:
        rows.append(f"| {row['scenario']} | {row['turns']} | {row['mutations']} | "
                    f"{row['nudges']} | {row['verified']} | {row['outcome']} |")
    for row in results['recall']:
        rows.append(f"| {row['distractors']} | {row['entries']} | {row['top1']}/{row['queries']} | "
                    f"{row['top3']}/{row['queries']} | {row['search_ms']:.2f} ms |")
    for row in results['compaction']:
        rows.append(f"| {row['turns']} | {row['summary']} | {row['messages_before']} | "
                    f"{row['messages_after']} | {row['removed']} | "
                    f"{row['facts_in_context']}/{row['facts']} | {row['facts_archived']} | "
                    f"{row['recall_top1']}/{row['asked']} | {row['recall_top3']}/{row['asked']} | "
                    f"{row['compact_ms']:.2f} |")
    for row in results['compaction_repeat']:
        reach = ' -> '.join(f"{value}/{row['early_facts']}"
                            for value in row['reachable_after_each_pass'])
        sizes = ', '.join(str(size) for size in row['archive_entries'])
        rows.append(f"| {row['turns']} | {row['passes']} | {row['early_facts']} | {reach} | {sizes} |")
    for row in results['vector']:
        rows.append(f"| {row['distractors']} | {row['entries']} | {row['archive_texts']} | "
                    f"{row['archive_provider_calls']} | {row['cold_query_ms']:.2f} | "
                    f"{row['repeat_texts']} | {row['repeat_provider_calls']} | "
                    f"{row['warm_query_ms']:.2f} |")
        rows.append(f"| {row['distractors']} | {row['lexical_ms']:.2f} | {row['vector_ms']:.2f} | "
                    f"{row['hybrid_ms']:.2f} | {row['top1_lexical']}/{row['queries']} | "
                    f"{row['top1_vector']}/{row['queries']} | {row['top1_hybrid']}/{row['queries']} |")
    for row in results['policy']:
        tool, _, argument = row['call'].partition(': ')
        rows.append(f"| `{tool}` with `{argument}` | {'yes' if row['ok'] else 'no'} | "
                    f"{'`' + row['tag'] + '`' if row['tag'] != 'success' else 'success'} |")
    for row in results['exposure']:
        rows.append(f"| {row['external']} | {row['budget']} | {row['exposed']} | {row['hidden']} | "
                    f"{row['schema_chars']:,} | {row['est_tokens']:,} | {row['vs_no_budget']:.2f}x |")
    for row in results['recovery']:
        rows.append(f"| `{row['wanted']}` | {'yes' if row['hidden_before'] else 'no'} | "
                    f"{row['budget']} | {len(row['exposed_before'])} | "
                    f"{'yes' if row['recovered'] else 'no'} |")
    return rows


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
