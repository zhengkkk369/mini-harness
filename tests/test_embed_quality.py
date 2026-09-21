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

COMMON = {'the', 'is', 'it', 'on', 'in', 'of', 'to', 'a', 'and', 'does', 'do', 'are', 'how',
          'what', 'when', 'where', 'which', 'many', 'times', 'long', 'live', 'name'}


# ------------------------------------------------------------------ the corpus


def test_every_query_is_a_paraphrase_rather_than_a_lookup():
    """The corpus exists because lexical scoring fails this shape.

    A query and its fact must share no rare token -- only ordinary words -- or
    the measurement would be another lexical lookup dressed up.
    """
    for query, fact in embed_quality.PARAPHRASES:
        shared = set(tokens(query)) & set(tokens(fact)) - COMMON
        assert shared <= COMMON, f'{query!r} and its fact share {sorted(shared)}'


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


# ------------------------------------------------------------------ labelling


def test_an_offline_run_says_the_numbers_are_the_stand_in():
    report = embed_quality.render(embed_quality.measure(Config(), count=10, offline=True),
                                  offline=True, cost_note='no provider')

    assert 'stand-in' in report
    assert 'say nothing about an embedding model' in report


def test_a_provider_run_does_not_claim_to_be_offline():
    report = embed_quality.render(embed_quality.measure(Config(), count=10, offline=True),
                                  offline=False, cost_note='provider: https://embed.test')

    assert 'stand-in' not in report
    assert 'provider: https://embed.test' in report


def test_the_report_has_a_row_per_backend():
    report = embed_quality.render(embed_quality.measure(Config(), count=10, offline=True),
                                  offline=True, cost_note='x')

    for backend in embed_quality.BACKENDS:
        assert f'| {backend} |' in report


# ------------------------------------------------------------------ the provider


def test_no_provider_is_detected(cfg_factory):
    assert embed_quality.provider_configured(cfg_factory(embed_base_url='')) is False


def test_a_configured_provider_is_detected(cfg_factory):
    cfg = cfg_factory(embed_base_url='https://embed.test/v1')

    assert embed_quality.provider_configured(cfg) is True


def test_the_environment_can_configure_the_provider(monkeypatch, cfg_factory):
    monkeypatch.setenv('MINI_HARNESS_EMBED_BASE_URL', 'https://embed.test/v1')

    assert embed_quality.provider_configured(cfg_factory(embed_base_url='')) is True


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
