"""Keep EXPERIMENTS.md's numbers identical to the recorded run.

The document quotes EXPERIMENTS.json, and `tests/test_experiments_doc.py` fails
if they disagree. Re-recording moves every timed row at once, so this both builds
the expected rows (the test imports `expected_rows`) and rewrites them in place:

    uv run python -m bench.experiments --repeats 15
    uv run python -m bench.sync_experiments_doc

One source for the document's row formats, used by the checker and the fixer, so
a format change cannot make one of them wrong about the other.
"""

import json
import sys

from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'EXPERIMENTS.md'
ARTIFACT = ROOT / 'EXPERIMENTS.json'

# The document's tables in the order they appear. Two of them share a header
# ("injected latency" starts both the parallel and the subagent table), so the
# header alone is not enough to place a row: the order is.
TABLES = ('parallel', 'subagents', 'trace_medians', 'trace_overhead', 'verify', 'recall',
          'compaction', 'compaction_repeat', 'vector_cost', 'vector_quality', 'policy',
          'exposure', 'recovery')

# Each table is recognised by any of these headers, the first being the one this
# module writes. The document styles some of them differently ("Allowed" reads
# better than "ok"), and a table that stopped being recognisable would go
# unchecked.
HEADERS = {
    'parallel': ('| Injected latency | Calls | Serial (median) | Concurrent (median) | Speedup |',),
    'subagents': ('| Injected latency | Calls | Serial (median) | Concurrent (median) | Speedup |',),
    'trace_medians': ('| Trace | Turns | Events | Median | Min | Max | Bytes written |',),
    'trace_overhead': ('| Turns | Events | Overhead (paired median) | Min pair | Max pair | Per event |',),
    'verify': ('| scenario | turns | mutations | verified | verdict | nudges | outcome |',
               '| Scenario | Turns | Mutations | Nudges | `verified` | Outcome |'),
    'recall': ('| Distractors | Archived messages | Top-1 | Top-3 | Search (median) |',),
    'compaction': ('| Turns | Summariser | Messages before | After | Removed | Facts in context | Archived | Recall top-1 | Recall top-3 | Compact (ms) |',),
    'compaction_repeat': ('| Turns | Passes | Early facts | Reachable after each pass | Archive entries |',),
    'vector_cost': ('| Distractors | Archived | Texts embedded | Provider calls | Cold query (ms) | Repeat texts | Repeat calls | Warm query (ms) |',),
    'vector_quality': ('| Distractors | Lexical (ms) | Vector (ms) | Hybrid (ms) | Top-1 lexical | Top-1 vector | Top-1 hybrid |',),
    'policy': ('| call | ok | tag |', '| Call | Allowed | Tag |'),
    'exposure': ('| bridged tools | budget | exposed | hidden | schema chars | est. tokens | vs no budget |',),
    'recovery': ('| wanted | hidden before | budget | exposed before | recovered |',),
}

def _key(line: str) -> str:
    """Headers are matched without case: the document styles some of them."""
    return ' '.join(line.split()).lower()

HEADERS_BY_KEY = {}
for _table, _spellings in HEADERS.items():
    for _spelling in _spellings:
        HEADERS_BY_KEY.setdefault(_key(_spelling), []).append(_table)
del _table, _spelling



def render_rows(results) -> dict:
    """{table: [row, ...]} -- every data row the document has to carry."""
    tables = {}
    tables['parallel'] = [
        f"| {row['latency_ms']:.0f} ms | {row['calls']} | {row['serial']['median_ms']:.2f} ms | "
        f"{row['parallel']['median_ms']:.2f} ms | {row['speedup']:.2f}x |"
        for row in results.get('parallel', [])]
    tables['subagents'] = [
        f"| {row['latency_ms']:.0f} ms | {row['calls']} | {row['serial']['median_ms']:.2f} ms | "
        f"{row['parallel']['median_ms']:.2f} ms | {row['speedup']:.2f}x |"
        for row in results.get('subagents', [])]
    tables['trace_medians'] = [
        f"| {row['trace']} | {row['turns']} | {row['events']} | {row['median_ms']:.2f} ms | "
        f"{row['min_ms']:.2f} ms | {row['max_ms']:.2f} ms | {row['bytes']:,} |"
        for row in results.get('trace', []) if 'median_ms' in row]
    tables['trace_overhead'] = [
        f"| {row['turns']} | {row['events']} | **{row['delta_ms']:+.2f} ms** | "
        f"{row['min_delta_ms']:+.2f} ms | {row['max_delta_ms']:+.2f} ms | {row['per_event_us']:.1f} us |"
        for row in results.get('trace', []) if 'per_event_us' in row]
    tables['verify'] = [
        f"| {row['scenario']} | {row['turns']} | {row['mutations']} | {row['verified']} | "
        f"{row['verification']} | {row['nudges']} | {row['outcome']} |"
        for row in results.get('verify', [])]
    tables['recall'] = [
        f"| {row['distractors']} | {row['entries']} | {row['top1']}/{row['queries']} | "
        f"{row['top3']}/{row['queries']} | {row['search_ms']:.2f} ms |"
        for row in results.get('recall', [])]
    tables['compaction'] = [
        f"| {row['turns']} | {row['summary']} | {row['messages_before']} | {row['messages_after']} | "
        f"{row['removed']} | {row['facts_in_context']}/{row['facts']} | {row['facts_archived']} | "
        f"{row['recall_top1']}/{row['asked']} | {row['recall_top3']}/{row['asked']} | "
        f"{row['compact_ms']:.2f} |"
        for row in results.get('compaction', [])]
    tables['compaction_repeat'] = [
        f"| {row['turns']} | {row['passes']} | {row['early_facts']} | "
        + ' -> '.join(f"{value}/{row['early_facts']}"
                      for value in row['reachable_after_each_pass'])
        + f" | {', '.join(str(size) for size in row['archive_entries'])} |"
        for row in results.get('compaction_repeat', [])]
    tables['vector_cost'] = [
        f"| {row['distractors']} | {row['entries']} | {row['archive_texts']} | "
        f"{row['archive_provider_calls']} | {row['cold_query_ms']:.2f} | {row['repeat_texts']} | "
        f"{row['repeat_provider_calls']} | {row['warm_query_ms']:.2f} |"
        for row in results.get('vector', [])]
    tables['vector_quality'] = [
        f"| {row['distractors']} | {row['lexical_ms']:.2f} | {row['vector_ms']:.2f} | "
        f"{row['hybrid_ms']:.2f} | {row['top1_lexical']}/{row['queries']} | "
        f"{row['top1_vector']}/{row['queries']} | {row['top1_hybrid']}/{row['queries']} |"
        for row in results.get('vector', [])]
    tables['policy'] = [
        f"| `{row['call'].partition(': ')[0]}` with `{row['call'].partition(': ')[2]}` | "
        f"{'yes' if row['ok'] else 'no'} | "
        f"{'`' + row['tag'] + '`' if row['tag'] != 'success' else 'success'} |"
        for row in results.get('policy', [])]
    tables['exposure'] = [
        f"| {row['external']} | {row['budget']} | {row['exposed']} | {row['hidden']} | "
        f"{row['schema_chars']:,} | {row['est_tokens']:,} | {row['vs_no_budget']:.2f}x |"
        for row in results.get('exposure', [])]
    tables['recovery'] = [
        f"| `{row['wanted']}` | {'yes' if row['hidden_before'] else 'no'} | {row['budget']} | "
        f"{len(row['exposed_before'])} | {'yes' if row['recovered'] else 'no'} |"
        for row in results.get('recovery', [])]
    return tables

def expected_rows(results) -> list:
    """Every data row, in the order the document presents them."""
    tables = render_rows(results)
    return [row for table in TABLES for row in tables[table]]

def sync(doc: Path = DOC, artifact: Path = ARTIFACT) -> tuple:
    """Rewrite the document's data rows from the artifact. Returns (changed, unplaced)."""
    results = json.loads(Path(artifact).read_text(encoding='utf-8'))
    tables = render_rows(results)
    pending = {table: deque(rows) for table, rows in tables.items()}
    lines = Path(doc).read_text(encoding='utf-8').splitlines()
    claimed = {key: deque(tables) for key, tables in HEADERS_BY_KEY.items()}
    table = None
    changed = 0
    unplaced = []
    for index, line in enumerate(lines):
        key = _key(line)
        if key in claimed and line.startswith('|'):
            table = claimed[key].popleft() if claimed[key] else None
            continue
        if not line.startswith('|'):
            table = None
            continue
        if table is None or set(line) <= set('|- :'):
            continue
        if pending[table]:
            if lines[index] != pending[table][0]:
                changed += 1
            lines[index] = pending[table].popleft()
        else:
            unplaced.append((table, line))
    Path(doc).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return changed, unplaced

def main() -> int:
    changed, unplaced = sync()
    print(f'rewrote {changed} row(s)')
    for table, line in unplaced:
        print(f'  unplaced [{table}]: {line}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
