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
# An error that means "this provider takes fewer inputs per request than you sent".
# Deliberately narrow: a bad key or a missing model must not be retried as a
# hundred tiny requests.
BATCH_LIMIT = re.compile(r'batch|too many|too large|maximum|at most', re.IGNORECASE)
# Most such errors state the ceiling ("should not be larger than 10"), and using
# it beats halving: halving 5 against a ceiling of 2 lands on 1 and pays twice
# the requests for the rest of the call.
STATED_LIMIT = re.compile(
    r'not be larger than\s+(\d+)|maximum (?:of |is )?(\d+)|at most\s+(\d+)', re.IGNORECASE)

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

    Providers do not agree on how many inputs one request may carry, and they do
    not advertise it either. The first endpoint this ran against, a
    ``text-embedding-v4`` deployment, refused anything above ten with a 400
    naming the batch. A cap is the provider's business, so a rejected chunk is
    halved and the smaller size is kept for the rest of the call (see
    ``_embed_chunk``); ``requests`` counts what that actually cost.
    """

    def __init__(self, model: str, api_key: str, base_url: str, batch: int = 96,
                 client = None) -> None:
        super().__init__(model)
        self.batch = max(1, batch)
        self.requests = 0
        self.client = client if client is not None else OpenAI(
            api_key = api_key, base_url = base_url or None, max_retries = 0)

    def _embed_many(self, texts: list) -> list:
        """One vector per text, in order.

        The consumption is a while loop rather than a `range` over `self.batch`,
        because `self.batch` can shrink *inside* the loop when a provider rejects
        a chunk: a precomputed stride then slices past texts that were never
        embedded, which is a missing cache entry and a `KeyError` later in the
        run. A chunk that was accepted whole -- or split internally and accepted
        in pieces -- covers exactly its own inputs.
        """
        vectors = []
        pending = list(texts)
        while pending:
            chunk = pending[:self.batch]
            vectors.extend(self._embed_chunk(chunk))
            pending = pending[len(chunk):]
        return vectors

    def _embed_chunk(self, chunk: list) -> list:
        """One request, split when the provider caps the batch lower.

        Only an error that names a size or batch problem is treated this way: any
        other failure (a bad key, a missing model, a rate limit) has to surface
        rather than be retried as a hundred tiny requests. The ceiling the error
        states is used when it states one, and the chunk is halved when it does
        not.
        """
        try:
            response = self.client.embeddings.create(model = self.model, input = chunk)
            self.requests += 1
        except Exception as error:
            if len(chunk) == 1 or not BATCH_LIMIT.search(str(error)):
                raise
            self.batch = min(self.batch, self._smaller(chunk, error))
            return self._embed_chunk(chunk[:self.batch]) + self._embed_chunk(chunk[self.batch:])
        ordered = sorted(response.data, key = lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    @staticmethod
    def _smaller(chunk: list, error: Exception) -> int:
        """The smaller batch size to retry with, from the error or by halving."""
        stated = [int(value) for group in STATED_LIMIT.findall(str(error)) for value in group if value]
        if stated and 0 < min(stated) < len(chunk):
            return min(stated)
        return max(1, len(chunk) // 2)

    def describe(self) -> dict:
        return {**super().describe(), 'embed_requests': self.requests}

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
