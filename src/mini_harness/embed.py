"""Embedders for the recall tool's vector backends.

Retrieval used to be lexical only: token overlap weighted by inverse document
frequency. That is exact, free and dependency-free, but it cannot match a query
to an archived message that says the same thing in different words -- ask about
"the maintenance window" and a message about "the deploy window" shares no rare
token with it.

Embeddings fix that class of miss, at a cost: an API call per text, the
conversation content leaves the machine, and the answer depends on a model
rather than on arithmetic. So the backend is pluggable and lexical stays the
default:

    lexical  no embedder at all (``build_embedder`` returns ``None``)
    hash     deterministic local vectors, for tests and for exercising the
             vector path without a provider. Same-word overlap only -- it is
             plumbing, not semantics.
    vector   embeddings only
    hybrid   lexical ranking fused with vector ranking

``Embedder`` owns the cache and the batching, so a subclass only has to
implement ``_embed_many`` against its provider. The cache is keyed by the text
itself and lives for the process, which matters because the archive is
re-embedded only when it grows, and repeated queries embed nothing at all.
"""

import hashlib
import math
import re

from openai import OpenAI

from mini_harness.config import CONFIG

HASH_DIM = 256
TOKEN = re.compile(r'[A-Za-z0-9_]+')

class Embedder:
    """A cache and a batch boundary in front of a provider's embed call."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._cache: dict = {}
        self.calls = 0
        self.hits = 0
        return

    def embed(self, texts) -> list:
        """Unit vectors for ``texts``, in order, embedding only what is not cached.

        Vectors are L2-normalised on the way into the cache, which is what makes
        a later similarity a plain dot product instead of a pair of square roots
        per comparison.
        """
        wanted = list(texts)
        missing = []
        for text in wanted:
            if text in self._cache:
                self.hits += 1
            elif text not in missing:
                missing.append(text)
        if missing:
            self.calls += 1
            for text, vector in zip(missing, self._embed_many(missing)):
                self._cache[text] = normalize(vector)
        return [self._cache[text] for text in wanted]

    def embed_one(self, text: str) -> list:
        return self.embed([text])[0]

    def _embed_many(self, texts: list) -> list:
        raise NotImplementedError

    def describe(self) -> dict:
        return {'backend_model': self.model, 'embed_calls': self.calls, 'embed_hits': self.hits}

def normalize(vector) -> list:
    """The unit vector in the same direction, or the zero vector."""
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]

def dot(left, right) -> float:
    """Similarity of two vectors that are already unit length."""
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))

def cosine(left, right) -> float:
    """Cosine similarity of arbitrary vectors, with a zero vector scoring zero."""
    if not left or not right or len(left) != len(right):
        return 0.0
    return dot(normalize(left), normalize(right))

class HashEmbedder(Embedder):
    """Deterministic vectors from hashed tokens. Offline, and not semantic.

    Each token lands in one of ``HASH_DIM`` slots, so two texts are similar when
    they share words -- the same signal lexical ranking already uses. It exists
    so the vector code path, the cache and the cosine ranking can be tested and
    demonstrated without a network call, and so it is honest to say the ranking
    quality of the real backends is not measured here.
    """

    def __init__(self, model: str = 'hash-local') -> None:
        super().__init__(model)
        return

    def _embed_many(self, texts: list) -> list:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list:
        vector = [0.0] * HASH_DIM
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode('utf-8'), digest_size = 8).digest()
            index = int.from_bytes(digest, 'big') % HASH_DIM
            vector[index] += 1.0
        return vector

class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible ``/embeddings`` endpoint.

    Note that the default provider for this harness does not serve one: the
    DeepSeek API has no embeddings route, so vector recall needs
    ``embed_base_url`` (and usually ``embed_api_key``) pointed at a provider
    that does.
    """

    def __init__(self, model: str, api_key: str, base_url: str, batch: int = 96,
                 client = None) -> None:
        super().__init__(model)
        self.batch = max(1, batch)
        self.client = client if client is not None else OpenAI(
            api_key = api_key, base_url = base_url or None, max_retries = 0)

    def _embed_many(self, texts: list) -> list:
        vectors = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start:start + self.batch]
            response = self.client.embeddings.create(model = self.model, input = chunk)
            ordered = sorted(response.data, key = lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        return vectors

def build_embedder(cfg = CONFIG):
    """The embedder for this configuration, or ``None`` for lexical recall.

    Built lazily, when the recall tool runs, so a run that never recalls pays
    nothing for the vector backends.
    """
    backend = cfg.recall_backend
    if backend == 'lexical':
        return None
    if backend == 'hash':
        return HashEmbedder()
    return OpenAIEmbedder(
        cfg.embed_model,
        cfg.embed_api_key or cfg.api_key,
        cfg.embed_base_url or cfg.base_url,
        cfg.embed_batch,
    )
