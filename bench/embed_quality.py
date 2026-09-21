"""Measure retrieval quality against a real embeddings provider.

Everything else in this repository runs offline, and so does this -- but only
against the token-hashing stand-in, which is a weaker lexical scorer rather than
a semantic one. The measurement that needs a provider is the one thing here that
cannot run in CI:

    export MINI_HARNESS_EMBED_BASE_URL=https://api.openai.com/v1
    export MINI_HARNESS_EMBED_API_KEY=sk-...
    export MINI_HARNESS_EMBED_MODEL=text-embedding-3-small
    uv run python -m bench.embed_quality --offline false

With no provider it runs the same code against the stand-in and says so in the
output, which still exercises the plumbing end to end (batching, the cache, the
fusion, the scan) and is what the tests use.

The corpus is paraphrase-shaped on purpose: each query and the fact that answers
it mean the same thing without sharing a rare word, which is exactly the case
lexical scoring cannot serve. Distractors share vocabulary with the queries so a
match is not free.

What this does not measure, even with a provider: whether retrieval *helps the
agent*. That is a task-level question and belongs with the model evaluation.
"""

import argparse
import json
import os
import statistics
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mini_harness.config import Config  # noqa: E402
from mini_harness.embed import HashEmbedder, build_embedder  # noqa: E402
from mini_harness.memory import Memory  # noqa: E402

BACKENDS = ('lexical', 'vector', 'hybrid')

# (query, the fact that answers it). The two share no content word at all, so
# lexical scoring has nothing to match on but "the" and "is". Checked by
# tests/test_embed_quality.py, which is how four of these were caught sharing a
# word they were not supposed to.
PARAPHRASES = [
    ('when is the maintenance period', 'the deploy window is 02:00 to 04:00 UTC on weekdays'),
    ('what is the name of the credential', 'the service token is stored in SERVICE_TOKEN'),
    ('which socket is it bound to', 'the service listens on port 8443'),
    ('how often may it try again', 'the retry budget is five attempts per request'),
    ('where does the data live', 'the database host is db.internal'),
    ('how long do cached entries live', 'the cache is invalidated every 15 minutes'),
]
FILLER = ('config cache worker queue schema index buffer handler parser session timeout '
          'retry logger metric deploy rollout cluster shard replica window service token '
          'port database credential listen retry').split()


def distractors(count: int) -> list:
    """Deterministic filler that shares vocabulary with the queries."""
    return [f'{FILLER[i % len(FILLER)]} {FILLER[(i * 7) % len(FILLER)]} '
            f'{FILLER[(i * 13) % len(FILLER)]} entry {i}' for i in range(count)]


def build_journal(path: Path, count: int) -> Path:
    removed = [{'role': 'tool', 'content': text} for text in distractors(count)]
    removed.extend({'role': 'user', 'content': fact} for _, fact in PARAPHRASES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'ts': 1.0, 'removed': removed}) + '\n', encoding='utf-8')
    return path


def embedder_for(backend: str, cfg: Config, offline: bool):
    """The embedder for a backend, or None for lexical."""
    if backend == 'lexical':
        return None
    if offline:
        return HashEmbedder()
    return build_embedder(Config(**{**cfg.__dict__, 'recall_backend': backend}))


def measure(cfg: Config, backends = BACKENDS, count: int = 200, offline: bool = False) -> list:
    """top-1/top-3, embedding calls and latency for each backend."""
    journal = build_journal(Path('.experiments') / 'embed-quality' / 'journal.jsonl', count)
    rows = []
    for backend in backends:
        memory = Memory(journal)
        embedder = embedder_for(backend, cfg, offline)
        top1 = top3 = 0
        samples = []
        for query, fact in PARAPHRASES:
            started = time.perf_counter()
            hits = memory.search(query, limit = 3, embedder = embedder,
                                 mode = 'lexical' if backend == 'lexical' else backend)
            samples.append((time.perf_counter() - started) * 1000)
            if hits and hits[0].text == fact:
                top1 += 1
            if any(hit.text == fact for hit in hits):
                top3 += 1
        calls = getattr(embedder, 'calls', 0)
        rows.append({
            'backend': backend,
            'model': getattr(embedder, 'model', 'none'),
            'entries': len(memory.entries()),
            'queries': len(PARAPHRASES),
            'top1': top1,
            'top3': top3,
            'embed_calls': calls,
            'embed_texts': sum(1 for _ in ()) or (len(memory.entries()) + len(PARAPHRASES)
                                                  if calls else 0),
            'median_ms': statistics.median(samples),
        })
    return rows


def provider_configured(cfg: Config) -> bool:
    """Whether a real embedding endpoint has been pointed at."""
    return bool(cfg.embed_base_url or os.environ.get('MINI_HARNESS_EMBED_BASE_URL'))


def render(rows: list, offline: bool, cost_note: str) -> str:
    lines = ['# retrieval quality', '']
    if offline:
        lines += ['**Offline run: the embedder is the token-hashing stand-in, which is a weaker '
                  'lexical scorer, not a semantic one. These numbers say nothing about an embedding '
                  'model; they exercise the plumbing.**', '']
    lines += ['| backend | model | archived | top-1 | top-3 | embed calls | median (ms) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['backend']} | {row['model']} | {row['entries']} | "
                     f"{row['top1']}/{row['queries']} | {row['top3']}/{row['queries']} | "
                     f"{row['embed_calls']} | {row['median_ms']:.2f} |")
    lines += ['', cost_note]
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument('--out', default='EMBED_QUALITY.json')
    parser.add_argument('--distractors', type=int, default=200)
    parser.add_argument('--offline', default='auto', choices = ('auto', 'true', 'false'),
                        help='auto uses the provider when one is configured, the stand-in otherwise')
    args = parser.parse_args()

    cfg = Config()
    offline = args.offline == 'true' or (args.offline == 'auto' and not provider_configured(cfg))
    rows = measure(cfg, count = args.distractors, offline = offline)
    note = ('No provider configured: the numbers are the stand-in. Point '
            'MINI_HARNESS_EMBED_BASE_URL at an /embeddings endpoint to measure a model.'
            if offline else
            f"Provider: {cfg.embed_base_url or os.environ.get('MINI_HARNESS_EMBED_BASE_URL')}, "
            f"model {cfg.embed_model}.")
    report = render(rows, offline, note)
    print(report)
    Path(args.out).write_text(json.dumps({'offline': offline,
                                          'provider': cfg.embed_base_url or None,
                                          'model': cfg.embed_model,
                                          'distractors': args.distractors,
                                          'rows': rows}, indent = 2) + '\n', encoding='utf-8')
    print(f'raw results written to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
