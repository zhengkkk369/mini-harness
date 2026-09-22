"""The retrieval-quality script, and the corpus it exists for.

The offline suite can compare lexical scoring with a token-hashing stand-in, and
that is all. Measuring a real embedding model needs a provider, which is why this
script is separate from the offline experiments -- but its plumbing, its corpus
and its labelling are testable without one.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import embed_quality  # noqa: E402
from mini_harness.config import Config  # noqa: E402
from mini_harness.memory import tokens  # noqa: E402

COMMON = {
    "the", "is", "it", "on", "in", "of", "to", "a", "an", "and", "or", "does", "do", "did",
    "are", "was", "were", "be", "been", "being", "how", "what", "when", "where", "which",
    "who", "why", "many", "much", "times", "long", "live", "name", "there", "that", "this",
    "these", "those", "their", "they", "them", "he", "she", "his", "her", "its", "i", "my",
    "me", "we", "us", "our", "you", "your", "for", "from", "by", "with", "at", "as", "if",
    "not", "no", "may", "must", "can", "could", "will", "would", "should", "shall", "might",
    "has", "have", "had", "am", "per", "each", "every", "some", "any", "all", "both", "few",
    "more", "most", "other", "same", "own", "too", "very", "also", "only", "just", "still",
    "yet", "even", "well", "about", "after", "before", "again", "over", "up", "down", "out",
    "off", "here", "now", "then", "than", "so", "into",
}


# ------------------------------------------------------------------ the corpus


def test_every_query_is_a_paraphrase_rather_than_a_lookup():
    """The corpus exists because lexical scoring fails this shape.

    A query and its fact must share no rare token -- only ordinary words -- or
    the measurement would be another lexical lookup dressed up.
    """
    for query, fact in embed_quality.PARAPHRASES:
        shared = set(tokens(query)) & set(tokens(fact)) - COMMON
        assert shared <= COMMON, f'{query!r} and its fact share {sorted(shared)}'


def test_the_corpus_is_large_enough_to_read_a_gap():
    """Six queries made one query 17 points, which is why it was extended.

    Forty puts a query at 2.5 points, so the ten-point gap between lexical and
    vector scoring is a gap rather than a coin flip.
    """
    assert len(embed_quality.PARAPHRASES) >= 40


def test_the_facts_are_distinguishable():
    facts = [fact for _, fact in embed_quality.PARAPHRASES]

    assert len(set(facts)) == len(facts)


def test_the_distractors_are_unique_and_share_the_vocabulary():
    made = embed_quality.distractors(80)

    assert len(set(made)) == 80, 'duplicates would make the archive smaller than it looks'
    assert any('deploy' in text for text in made), 'the filler should overlap the queries'


def test_the_journal_is_one_record_of_everything(tmp_path):
    path = embed_quality.build_journal(tmp_path / 'journal.jsonl', 20)

    record = json.loads(path.read_text(encoding='utf-8'))

    assert len(record['removed']) == 20 + len(embed_quality.PARAPHRASES)


# ------------------------------------------------------------------ measuring


def test_every_backend_is_measured_offline():
    rows = embed_quality.measure(Config(), count=20, offline=True)

    assert [row['backend'] for row in rows] == ['lexical', 'vector', 'hybrid']
    for row in rows:
        assert row['queries'] == len(embed_quality.PARAPHRASES)
        assert 0 <= row['top1'] <= row['queries']
        assert row['top3'] >= row['top1']
        assert row['entries'] == 20 + len(embed_quality.PARAPHRASES)


def test_lexical_embeds_nothing_and_the_others_do():
    rows = {row['backend']: row for row in embed_quality.measure(Config(), count=20, offline=True)}

    assert rows['lexical']['embed_calls'] == 0
    assert rows['lexical']['model'] == 'none'
    assert rows['vector']['embed_calls'] >= 1
    assert rows['hybrid']['embed_calls'] >= 1


def test_measuring_the_same_thing_twice_gives_the_same_answer():
    first = embed_quality.measure(Config(), count=20, offline=True)
    second = embed_quality.measure(Config(), count=20, offline=True)

    assert [(row['backend'], row['top1'], row['top3']) for row in first] == \
           [(row['backend'], row['top1'], row['top3']) for row in second]


def test_the_first_query_is_reported_separately_from_the_rest():
    """One median over six samples would report the cold start as a query cost.

    The first query embeds the whole archive; every later one embeds the query
    alone and reads cached vectors.
    """
    rows = embed_quality.measure(Config(), count=50, offline=True)

    for row in rows:
        assert row['cold_ms'] >= 0
        assert row['warm_median_ms'] >= 0
        assert row['median_ms'] >= 0


def scenario(rows, **overrides):
    return {'distractors': overrides.pop('distractors', 10), 'rows': rows}


# ------------------------------------------------------------------ labelling


def test_an_offline_run_says_the_numbers_are_the_stand_in():
    report = embed_quality.render([scenario(embed_quality.measure(Config(), count=10, offline=True))],
                                  offline=True, cost_note='no provider')

    assert 'stand-in' in report
    assert 'say nothing about an embedding model' in report


def test_a_provider_run_does_not_claim_to_be_offline():
    report = embed_quality.render([scenario(embed_quality.measure(Config(), count=10, offline=True))],
                                  offline=False, cost_note='provider: https://embed.test')

    assert 'stand-in' not in report
    assert 'provider: https://embed.test' in report


def test_the_report_has_a_row_per_backend():
    report = embed_quality.render([scenario(embed_quality.measure(Config(), count=10, offline=True))],
                                  offline=True, cost_note='x')

    for backend in embed_quality.BACKENDS:
        assert f'| {backend} |' in report


def test_every_measured_archive_size_gets_its_own_rows():
    report = embed_quality.render(
        [scenario(embed_quality.measure(Config(), count=10, offline=True), distractors=10),
         scenario(embed_quality.measure(Config(), count=60, offline=True), distractors=60)],
        offline=True, cost_note='x')

    assert f"| {10 + len(embed_quality.PARAPHRASES)} | lexical |" in report
    assert f"| {60 + len(embed_quality.PARAPHRASES)} | lexical |" in report


def test_the_report_carries_the_request_count_beside_the_call_count():
    """A provider bills requests, and the batch ceiling decides how many."""
    rows = embed_quality.measure(Config(), count=10, offline=True)
    report = embed_quality.render([scenario(rows)], offline=True, cost_note='x')
    header = [line for line in report.splitlines() if line.startswith('| archived')][0]

    assert 'requests' in header
    assert all('provider_requests' in row for row in rows)


# ------------------------------------------------------------------ the provider


def test_no_provider_is_detected(cfg_factory):
    assert embed_quality.provider_configured(cfg_factory(embed_base_url='')) is False


def test_a_configured_provider_is_detected(cfg_factory):
    cfg = cfg_factory(embed_base_url='https://embed.test/v1')

    assert embed_quality.provider_configured(cfg) is True


def test_the_environment_can_configure_the_provider(monkeypatch, cfg_factory):
    """The documented command has to reach the embedder, not only the report.

    The first version built its configuration with a bare `Config()`, so
    `provider_configured` saw the environment variable and said "online" while
    the embedder was constructed with the embedding fields still empty -- which
    pointed it at the main model's base_url. This asserts the whole path.
    """
    from mini_harness.config import build_config
    from mini_harness.embed import OpenAIEmbedder

    monkeypatch.setenv('MINI_HARNESS_EMBED_BASE_URL', 'https://embed.test/v1')
    monkeypatch.setenv('MINI_HARNESS_EMBED_API_KEY', 'embed-key')
    monkeypatch.setenv('MINI_HARNESS_EMBED_MODEL', 'embed-small')

    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return type('C', (), {'embeddings': type('E', (), {
            'create': lambda self, **_: type('R', (), {'data': []})()})()})()

    monkeypatch.setattr('mini_harness.embed.OpenAI', fake_openai)
    cfg = build_config()

    assert embed_quality.provider_configured(cfg) is True
    embedder = embed_quality.embedder_for('vector', cfg, offline=False)

    assert isinstance(embedder, OpenAIEmbedder)
    assert seen['base_url'] == 'https://embed.test/v1'
    assert seen['api_key'] == 'embed-key'
    assert embedder.model == 'embed-small'


def test_a_local_config_file_can_configure_the_provider(monkeypatch, tmp_path):
    """The same path, configured the other way this tool supports."""
    from mini_harness.config import build_config

    path = tmp_path / 'config.yaml'
    path.write_text('embedding:\n  base_url: "https://embed.local/v1"\n'
                    '  model_name: "text-embedding-local"\n  api_key: "local-key"\n',
                    encoding='utf-8')
    monkeypatch.setenv('MINI_HARNESS_CONFIG_FILE', str(path))
    cfg = build_config()

    assert embed_quality.provider_configured(cfg) is True
    assert cfg.embed_base_url == 'https://embed.local/v1'
    assert cfg.embed_model == 'text-embedding-local'


def test_an_absent_local_file_means_no_provider(monkeypatch, tmp_path):
    from mini_harness.config import build_config

    monkeypatch.setenv('MINI_HARNESS_CONFIG_FILE', str(tmp_path / 'absent.yaml'))
    cfg = build_config()

    assert embed_quality.provider_configured(cfg) is False
    assert embed_quality.measure(cfg, backends = ('lexical',), count = 0, offline = True)


def test_a_provider_run_builds_a_real_embedder(cfg_factory, monkeypatch):
    """The online path must not silently fall back to the stand-in."""
    from mini_harness.embed import OpenAIEmbedder

    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return type('C', (), {'embeddings': type('E', (), {
            'create': lambda self, **_: type('R', (), {'data': []})()})()})()

    monkeypatch.setattr('mini_harness.embed.OpenAI', fake_openai)
    cfg = cfg_factory(recall_backend='vector', embed_base_url='https://embed.test/v1',
                      embed_api_key='k')

    embedder = embed_quality.embedder_for('vector', cfg, offline=False)

    assert isinstance(embedder, OpenAIEmbedder)
    assert seen['base_url'] == 'https://embed.test/v1'


def test_an_offline_vector_run_uses_the_stand_in():
    from mini_harness.embed import HashEmbedder

    assert isinstance(embed_quality.embedder_for('vector', Config(), offline=True), HashEmbedder)
    assert embed_quality.embedder_for('lexical', Config(), offline=True) is None
