"""Search what compaction removed.

Compaction is lossy: the summariser replaces a prefix of the conversation with
a few sentences. Until now the removed messages were only written to
``mini_harness_history.jsonl`` as an audit trail and never read again, so any
detail the summary dropped was gone for the rest of the run.

This module turns that journal into something queryable. The default retrieval
is plain lexical scoring -- token overlap weighted by inverse document frequency
-- so it needs no model, no embeddings and no extra dependency, and the same
query always returns the same answer. The ranking is pluggable: see
``mini_harness.embed`` for the vector backends and ``Memory.search`` for how the
two are combined.

The journal format is unchanged and stays compatible with what
``bench/atif.py`` reads back.
"""

import json
import math
import re

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mini_harness.config import CONFIG
from mini_harness.embed import dot

JOURNAL_NAME = 'mini_harness_history.jsonl'
TOKEN = re.compile(r'[A-Za-z0-9_]+')
# Reciprocal rank fusion, with the standard constant: it keeps the head of each
# ranking dominant without letting either list's raw scores dominate the other.
RRF_K = 60

def journal_for(session_path: str|Path|None, cfg = CONFIG) -> Path:
    """The journal belonging to a session file, or to the default one.

    Both the writer (compaction) and the reader (the recall tool) go through
    this, so they cannot disagree about the file name.
    """
    if session_path:
        base = Path(session_path)
    elif cfg.session_path:
        base = Path(cfg.session_path)
    else:
        base = cfg.work_space/'session.json'
    return base.with_name(JOURNAL_NAME)

def journal_path(cfg = CONFIG) -> Path:
    """Where the compaction journal lives for a run using this config."""
    return journal_for(cfg.session_path, cfg)

def tokens(text: str) -> list:
    return TOKEN.findall(text.lower())

def rank(terms: set, tokenised: list) -> list:
    """(index, score) for every document that matches, best first.

    Token overlap weighted by inverse document frequency, so a term appearing in
    one document outweighs one appearing in all of them. Shared by the journal
    search and the tool selector so both rank the same way.
    """
    if not terms or not tokenised:
        return []
    total = len(tokenised)
    document_frequency = {term: sum(1 for doc in tokenised if term in doc) for term in terms}
    scored = []
    for index, doc in enumerate(tokenised):
        counts = Counter(doc)
        score = 0.0
        for term in terms:
            frequency = counts.get(term, 0)
            if not frequency or not document_frequency[term]:
                continue
            score += (1 + math.log(frequency)) * math.log(1 + total / document_frequency[term])
        if score > 0:
            scored.append((index, score))
    scored.sort(key = lambda pair: (-pair[1], pair[0]))
    return scored

def _message_text(message: dict) -> tuple:
    """(role, searchable text) for one archived message."""
    role = str(message.get('role', '?'))
    parts = []
    content = message.get('content')
    if content:
        parts.append(str(content))
    for call in message.get('tool_calls') or []:
        function = call.get('function') or {}
        parts.append(f"{function.get('name', '?')} {function.get('arguments', '')}")
    return role, '\n'.join(parts)

@dataclass(frozen = True)
class MemoryEntry:
    index: int
    ts: float
    role: str
    text: str
    tokens: tuple

@dataclass(frozen = True)
class MemoryHit:
    score: float
    role: str
    ts: float
    index: int
    text: str

class Memory:
    """A read-only view over a compaction journal, re-read when the file grows."""

    def __init__(self, path: str|Path) -> None:
        self.path = Path(path)
        self._stamp = None
        self._entries: list = []
        return

    def entries(self) -> list:
        try:
            stat = self.path.stat()
        except OSError:
            self._stamp, self._entries = None, []
            return []
        stamp = (stat.st_mtime, stat.st_size)
        if stamp == self._stamp:
            return list(self._entries)
        self._entries = self._read()
        self._stamp = stamp
        return list(self._entries)

    def _read(self) -> list:
        entries = []
        try:
            raw = self.path.read_text(errors = 'replace', encoding = 'utf-8')
        except OSError:
            return entries
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            removed = record.get('removed') if isinstance(record, dict) else None
            if not isinstance(removed, list):
                continue
            ts = record.get('ts') or 0.0
            for message in removed:
                if not isinstance(message, dict):
                    continue
                role, text = _message_text(message)
                if not text.strip():
                    continue
                entries.append(MemoryEntry(len(entries), ts, role, text, tuple(tokens(text))))
        return entries

    def lexical_order(self, terms: set) -> list:
        """(index, score) for every entry sharing a term, best first."""
        return rank(terms, [entry.tokens for entry in self.entries()])

    def vector_order(self, query: str, embedder) -> list:
        """(index, similarity) for every entry, best first.

        The embedder hands back unit vectors, so the comparison is a dot
        product. Raises whatever the provider raises: the caller decides whether
        a failed embedding is worth falling back to lexical ranking for.
        """
        entries = self.entries()
        if not entries:
            return []
        vectors = embedder.embed([entry.text for entry in entries])
        query_vector = embedder.embed_one(query)
        scored = [(index, dot(query_vector, vector)) for index, vector in enumerate(vectors)]
        scored = [(index, score) for index, score in scored if score > 0]
        scored.sort(key = lambda pair: (-pair[1], pair[0]))
        return scored

    @staticmethod
    def fuse(orders) -> list:
        """Reciprocal rank fusion of several (index, score) rankings.

        Raw scores from two different ranking functions are not comparable, so
        only the positions are combined. An entry missing from one list simply
        scores nothing there.
        """
        totals: dict = {}
        for ordered in orders:
            for position, (index, _) in enumerate(ordered, start = 1):
                totals[index] = totals.get(index, 0.0) + 1.0 / (RRF_K + position)
        return sorted(totals.items(), key = lambda pair: (-pair[1], pair[0]))

    def search(self, query: str, limit: int = 5, role: str|None = None,
               embedder = None, mode: str = 'lexical') -> list:
        """Best matches first, ties broken by the order they were archived.

        ``mode`` is the configured backend: ``lexical`` (the default),
        ``vector``, or ``hybrid`` for the fusion of the two. The ``score`` on a
        hit is comparable only within one call, and only for the same mode.
        """
        entries = self.entries()
        terms = set(tokens(query))
        if not entries or not terms:
            return []
        if mode == 'lexical' or embedder is None:
            ordered = self.lexical_order(terms)
        elif mode == 'vector':
            ordered = self.vector_order(query, embedder)
        else:
            ordered = self.fuse([self.lexical_order(terms), self.vector_order(query, embedder)])
        hits = []
        for index, score in ordered:
            entry = entries[index]
            if role and entry.role != role:
                continue
            hits.append(MemoryHit(score, entry.role, entry.ts, entry.index, entry.text))
            if len(hits) >= max(0, limit):
                break
        return hits

    def stats(self) -> dict:
        entries = self.entries()
        return {
            'entries': len(entries),
            'bytes': self.path.stat().st_size if self.path.exists() else 0,
            'roles': dict(Counter(entry.role for entry in entries)),
        }
