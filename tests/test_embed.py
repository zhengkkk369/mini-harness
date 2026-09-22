"""The pluggable recall backend: embedders, fusion, and the fallback.

Nothing here talks to a provider. The offline ``hash`` embedder exists so the
vector path, the cache and the cosine ranking can be tested and measured
without a network call; a fake client covers the OpenAI-compatible path.
"""

import json
from types import SimpleNamespace

import pytest

from mini_harness import embed as embed_module
from mini_harness.config import RECALL_BACKENDS
from mini_harness.embed import (
    HASH_DIM, Embedder, HashEmbedder, OpenAIEmbedder, build_embedder, cosine, dot, normalize,
)
from mini_harness.memory import JOURNAL_NAME, Memory, journal_path
from mini_harness.tool import box
from mini_harness.trace import TRACE

from tests.test_memory import journal, message


# ------------------------------------------------------------------ cosine


def test_identical_vectors_are_maximally_similar():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_scaled_vectors_are_still_similar():
    assert cosine([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_disjoint_vectors_are_unrelated():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_a_zero_or_mismatched_vector_scores_zero_instead_of_nan():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([], []) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0


# ------------------------------------------------------------------ the hash embedder


def test_hash_vectors_are_deterministic_and_fixed_width():
    embedder = HashEmbedder()

    first = embedder.embed_one('the deploy window is 02:00')
    second = HashEmbedder().embed_one('the deploy window is 02:00')

    assert first == second
    assert len(first) == HASH_DIM


def test_hash_vectors_are_similar_when_the_words_are():
    embedder = HashEmbedder()

    assert cosine(embedder.embed_one('deploy window'), embedder.embed_one('deploy window 02:00')) > 0
    assert cosine(embedder.embed_one('deploy window'), embedder.embed_one('unrelated tokens')) == 0


def test_embed_keeps_the_order_of_its_inputs():
    embedder = HashEmbedder()

    vectors = embedder.embed(['alpha', 'beta'])

    assert vectors[0] == embedder.embed_one('alpha')
    assert vectors[1] == embedder.embed_one('beta')


def test_embedding_the_same_text_twice_calls_the_provider_once():
    class Counting(Embedder):
        def __init__(self):
            super().__init__('counting')
            self.texts = []

        def _embed_many(self, texts):
            self.texts.extend(texts)
            return [[1.0] for _ in texts]

    embedder = Counting()
    embedder.embed(['alpha', 'beta'])
    embedder.embed(['beta', 'gamma'])

    assert embedder.calls == 2
    assert embedder.texts == ['alpha', 'beta', 'gamma']
    assert embedder.hits == 1


def test_a_repeated_text_inside_one_call_is_embedded_once():
    class Counting(Embedder):
        def __init__(self):
            super().__init__('counting')
            self.texts = []

        def _embed_many(self, texts):
            self.texts.extend(texts)
            return [[1.0] for _ in texts]

    embedder = Counting()

    assert len(embedder.embed(['same', 'same'])) == 2
    assert embedder.texts == ['same']


def test_describe_reports_the_cache_behaviour():
    embedder = HashEmbedder()
    embedder.embed(['alpha'])
    embedder.embed(['alpha'])

    assert embedder.describe() == {'backend_model': 'hash-local', 'embed_calls': 1, 'embed_hits': 1}


def test_the_base_embedder_has_no_provider():
    with pytest.raises(NotImplementedError):
        Embedder('none').embed(['alpha'])


# ------------------------------------------------------------------ the provider embedder


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({'model': model, 'input': list(input)})
        # Deliberately out of order: the adapter has to reorder by index.
        data = [SimpleNamespace(index=i, embedding=[float(i), 1.0])
                for i in reversed(range(len(input)))]
        return SimpleNamespace(data=data)


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


def test_provider_vectors_come_back_in_the_order_they_were_asked_for():
    client = FakeClient()
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', client=client)

    vectors = embedder.embed(['a', 'b'])

    # The fake provider returns them reversed, and unnormalised.
    assert vectors[0] == [0.0, 1.0]
    assert vectors[1][0] == pytest.approx(vectors[1][1])
    assert sum(value * value for value in vectors[1]) == pytest.approx(1.0)


def test_the_cache_holds_unit_vectors():
    embedder = HashEmbedder()

    vector = embedder.embed_one('alpha beta gamma')

    assert sum(value * value for value in vector) == pytest.approx(1.0)


# ------------------------------------------- a provider's own batch ceiling


class CappedEmbeddings(FakeEmbeddings):
    """A provider that refuses more than ``cap`` inputs per request.

    Modelled on the message a real ``text-embedding-v4`` deployment returned:
    "Value error, batch size is invalid, it should not be larger than 10."
    """

    def __init__(self, cap: int, error: str = 'batch size is invalid, it should not be'
                                             ' larger than {cap}.'):
        super().__init__()
        self.cap = cap
        self.error = error

    def create(self, model, input):
        if len(input) > self.cap:
            self.calls.append({'model': model, 'input': list(input), 'rejected': True})
            raise ValueError(self.error.format(cap=self.cap))
        return super().create(model, input)


def test_a_batch_ceiling_is_discovered_instead_of_configured():
    """A provider's cap must not require the caller to know it in advance."""
    client = FakeClient()
    client.embeddings = CappedEmbeddings(cap=10)
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=96, client=client)

    vectors = embedder.embed([f'text {index}' for index in range(40)])

    assert len(vectors) == 40
    assert embedder.batch <= 10
    # Every text was embedded exactly once, in order, despite the rejects.
    accepted = [call for call in client.embeddings.calls if not call.get('rejected')]
    assert [text for call in accepted for text in call['input']] == [
        f'text {index}' for index in range(40)]
    assert embedder.requests == len(accepted)


def test_a_rejected_chunk_still_returns_vectors_for_its_texts():
    client = FakeClient()
    client.embeddings = CappedEmbeddings(cap=2)
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=8, client=client)

    vectors = embedder.embed(['a', 'b', 'c', 'd', 'e'])

    assert len(vectors) == 5
    assert embedder.batch == 2
    assert all(sum(value * value for value in vector) == pytest.approx(1.0)
               for vector in vectors)


def test_an_error_that_is_not_a_batch_ceiling_is_raised():
    """A bad key must fail, not be retried as a hundred tiny requests."""
    class Broken(FakeEmbeddings):
        def create(self, model, input):
            self.calls.append({'model': model, 'input': list(input)})
            raise ValueError('Incorrect API key provided')

    client = FakeClient()
    client.embeddings = Broken()
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=4, client=client)

    with pytest.raises(ValueError, match='Incorrect API key'):
        embedder.embed(['a', 'b', 'c'])

    assert len(client.embeddings.calls) == 1


def test_every_text_is_embedded_when_the_batch_shrinks_mid_call():
    """The stride must follow the shrinking batch, not the original one.

    With a precomputed stride the loop skipped the texts between the old and new
    boundaries: the chunk after the rejection started at the *old* offset, so a
    slice of the middle of the list was never embedded at all, and the missing
    cache entries surfaced as a `KeyError` in a later search. The sizes here are
    chosen so that stride and shrink disagree.
    """
    client = FakeClient()
    client.embeddings = CappedEmbeddings(cap=3)
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=20, client=client)
    texts = [f'text {index}' for index in range(25)]

    first = embedder.embed(texts)
    requests = embedder.requests
    # A complete cache means the second call costs nothing at all.
    second = embedder.embed(texts)

    assert len(first) == len(texts) == len(second)
    assert first == second
    assert embedder.requests == requests
    assert all(vector for vector in first)


def test_a_ceiling_that_is_not_stated_falls_back_to_halving():
    """Some providers reject without naming a number; that must still work."""
    client = FakeClient()
    client.embeddings = CappedEmbeddings(cap=10, error='too many inputs in one request')
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=96, client=client)

    vectors = embedder.embed([f'text {index}' for index in range(30)])

    assert len(vectors) == 30
    assert embedder.batch <= 10
    assert embedder.requests > 3


def test_a_single_text_that_is_rejected_is_not_split_forever():
    client = FakeClient()
    client.embeddings = CappedEmbeddings(cap=0)
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=4, client=client)

    with pytest.raises(ValueError):
        embedder.embed(['one'])


def test_the_batch_description_counts_requests_not_calls():
    """`embed_calls` counts embed() calls; the provider bills requests."""
    client = FakeClient()
    client.embeddings = CappedEmbeddings(cap=10)
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=96, client=client)

    embedder.embed([f'text {index}' for index in range(20)])
    described = embedder.describe()

    assert described['embed_calls'] == 1
    assert described['embed_requests'] == len(
        [call for call in client.embeddings.calls if not call.get('rejected')])
    assert described['embed_requests'] > 1


def test_normalising_a_zero_vector_keeps_it_finite():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_dot_is_the_similarity_of_unit_vectors():
    assert dot([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert dot([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert dot([1.0], [1.0, 2.0]) == 0.0


def test_provider_calls_are_chunked_by_the_batch_size():
    client = FakeClient()
    embedder = OpenAIEmbedder('m', 'key', 'https://example.test/v1', batch=2, client=client)

    embedder.embed(['a', 'b', 'c', 'd', 'e'])

    assert [len(call['input']) for call in client.embeddings.calls] == [2, 2, 1]
    assert all(call['model'] == 'm' for call in client.embeddings.calls)


# ------------------------------------------------------------------ building one from config


def test_lexical_configuration_needs_no_embedder(cfg):
    assert build_embedder(cfg) is None


def test_the_hash_backend_builds_the_offline_embedder(cfg_factory):
    embedder = build_embedder(cfg_factory(recall_backend='hash'))

    assert isinstance(embedder, HashEmbedder)


def test_the_vector_backend_builds_a_provider_embedder(cfg_factory, monkeypatch):
    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(embed_module, 'OpenAI', fake_openai)
    cfg = cfg_factory(recall_backend='vector', embed_model='embed-small', embed_api_key='embed-key',
                      embed_base_url='https://embed.test/v1', embed_batch=7)

    embedder = build_embedder(cfg)

    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == 'embed-small'
    assert embedder.batch == 7
    assert seen == {'api_key': 'embed-key', 'base_url': 'https://embed.test/v1', 'max_retries': 0}


def test_the_vector_backend_falls_back_to_the_main_credentials(cfg_factory, monkeypatch):
    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(embed_module, 'OpenAI', fake_openai)
    cfg = cfg_factory(recall_backend='vector')

    build_embedder(cfg)

    assert seen['api_key'] == cfg.api_key
    assert seen['base_url'] == cfg.base_url


def test_every_documented_backend_is_accepted():
    assert RECALL_BACKENDS == {'lexical', 'hash', 'vector', 'hybrid'}


# ------------------------------------------------------------------ the memory side


def archived(cfg, *texts):
    return journal(journal_path(cfg), {'ts': 1, 'removed': [message('user', text) for text in texts]})


def test_vector_search_ranks_by_cosine(cfg_factory):
    cfg = cfg_factory(recall_backend='hash')
    memory = Memory(archived(cfg, 'the deploy window is 02:00', 'unrelated tool output'))

    hits = memory.search('deploy window', embedder=HashEmbedder(), mode='vector')

    assert hits
    assert '02:00' in hits[0].text


def test_vector_search_returns_nothing_when_no_word_is_shared(cfg_factory):
    """The offline embedder has no semantics, and the test says so."""
    cfg = cfg_factory(recall_backend='hash')
    memory = Memory(archived(cfg, 'the deploy window is 02:00'))

    assert memory.search('zzzz', embedder=HashEmbedder(), mode='vector') == []


def test_hybrid_fuses_both_rankings(cfg_factory):
    cfg = cfg_factory(recall_backend='hybrid')
    memory = Memory(archived(cfg, 'needle in the haystack', 'unrelated filler'))

    fused = memory.search('needle', embedder=HashEmbedder(), mode='hybrid')
    lexical = memory.search('needle', mode='lexical')

    assert [hit.text for hit in fused] == [hit.text for hit in lexical]


def test_fusion_lets_each_ranking_promote_its_own_best():
    """Second in one list and first in the other is enough to win."""
    fused = Memory.fuse([[(0, 9.0), (1, 1.0)], [(1, 0.9), (2, 0.1)]])

    assert [index for index, _ in fused] == [1, 0, 2]
    assert fused[0][1] > fused[1][1]


def test_fusion_keeps_an_entry_that_only_one_ranking_found():
    fused = Memory.fuse([[(0, 5.0)], [(1, 0.5)]])

    assert {index for index, _ in fused} == {0, 1}


def test_hybrid_over_real_entries_orders_the_matches(cfg_factory):
    cfg = cfg_factory(recall_backend='hybrid')
    memory = Memory(archived(cfg, 'alpha beta', 'alpha gamma'))

    fused = memory.search('alpha gamma', embedder=HashEmbedder(), mode='hybrid')

    assert [hit.text for hit in fused] == ['alpha gamma', 'alpha beta']


def test_vector_search_honours_the_role_filter_and_the_limit(cfg_factory):
    cfg = cfg_factory(recall_backend='hash')
    path = journal(journal_path(cfg), {'ts': 1, 'removed': [
        message('user', 'alpha from the user'),
        message('tool', 'alpha from a tool'),
    ]})
    memory = Memory(path)

    assert [hit.role for hit in memory.search('alpha', role='tool', embedder=HashEmbedder(), mode='vector')] == ['tool']
    assert len(memory.search('alpha', limit=1, embedder=HashEmbedder(), mode='vector')) == 1


def test_a_missing_embedder_falls_back_to_lexical(cfg_factory):
    cfg = cfg_factory(recall_backend='vector')
    memory = Memory(archived(cfg, 'needle here'))

    assert memory.search('needle', embedder=None, mode='vector')


def test_the_entry_index_is_rebuilt_when_the_journal_grows_under_vector_search(cfg_factory):
    cfg = cfg_factory(recall_backend='hash')
    path = archived(cfg, 'alpha one')
    memory = Memory(path)
    embedder = HashEmbedder()
    assert len(memory.search('alpha', embedder=embedder, mode='vector')) == 1

    journal(path, {'ts': 2, 'removed': [message('user', 'alpha two')]})

    assert len(memory.search('alpha', embedder=embedder, mode='vector')) == 2


# ------------------------------------------------------------------ the tool


def test_recall_reports_the_backend_it_used(cfg_factory):
    cfg = cfg_factory(recall_backend='hash')
    archived(cfg, 'the deploy window is 02:00 to 04:00 UTC', 'unrelated')

    result = box.recall(box.RecallInput(query='deploy window'), cfg=cfg)

    assert '(hash)' in result
    assert '02:00 to 04:00' in result


def test_recall_can_be_switched_off_for_a_backend_too(cfg_factory):
    cfg = cfg_factory(recall_backend='hash', recall_enabled=False)
    archived(cfg, 'alpha')

    assert box.recall(box.RecallInput(query='alpha'), cfg=cfg) == '[recall]: disabled by configuration'


def test_recall_traces_the_backend_and_the_cache(cfg_factory, session_dir):
    cfg = cfg_factory(recall_backend='hash')
    archived(cfg, 'alpha detail')
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        box.recall(box.RecallInput(query='alpha'), cfg=cfg)
    finally:
        TRACE.configure(None)

    event = json.loads(path.read_text(encoding='utf-8').splitlines()[0])

    assert event['backend'] == 'hash'
    assert event['embed_calls'] >= 1
    assert event['backend_model'] == 'hash-local'


def test_a_provider_failure_falls_back_to_lexical_ranking(cfg_factory, monkeypatch):
    cfg = cfg_factory(recall_backend='vector')
    archived(cfg, 'the deploy window is 02:00 to 04:00 UTC')

    class Broken(Embedder):
        def _embed_many(self, texts):
            raise RuntimeError('no embeddings endpoint')

    monkeypatch.setattr(box, 'build_embedder', lambda cfg=None: Broken('broken'))

    result = box.recall(box.RecallInput(query='deploy window'), cfg=cfg)

    assert 'fell back to lexical' in result
    assert 'no embeddings endpoint' in result
    assert '02:00 to 04:00' in result


def test_the_fallback_is_traced_as_a_fallback(cfg_factory, session_dir, monkeypatch):
    cfg = cfg_factory(recall_backend='vector')
    archived(cfg, 'alpha detail')

    class Broken(Embedder):
        def _embed_many(self, texts):
            raise RuntimeError('boom')

    monkeypatch.setattr(box, 'build_embedder', lambda cfg=None: Broken('broken'))
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        box.recall(box.RecallInput(query='alpha'), cfg=cfg)
    finally:
        TRACE.configure(None)

    event = json.loads(path.read_text(encoding='utf-8').splitlines()[0])

    assert event['backend'] == 'lexical-fallback'
    assert event['hits'] == 1


def test_the_default_backend_is_lexical(cfg_factory):
    cfg = cfg_factory()
    archived(cfg, 'deploy window detail')

    assert '(lexical)' in box.recall(box.RecallInput(query='deploy'), cfg=cfg)
