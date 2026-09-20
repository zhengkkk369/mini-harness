"""Token, cost and wall-clock budgets for one agent run.

The agent checks `exceeded()` before each request. A run that stops on a budget
reports which one, so it is distinguishable from running out of turns.
"""

import time

from dataclasses import dataclass, field

STOP_TOKENS = 'tokens'
STOP_COST = 'cost'
STOP_WALL = 'wall'

PER_MILLION = 1_000_000

@dataclass
class Budget:
    """Accumulates usage and decides when a run must stop.

    ``price_in`` / ``price_out`` are US dollars per million tokens. Cost stays
    at zero unless both are set, so token-only budgets need no pricing data.

    ``price_cache_in`` is the rate for input the provider served from its cache.
    That rate can be a fraction of the uncached one, so leaving it unset is
    conservative rather than wrong: cached tokens are then billed at the full
    input price and the figure is an upper bound.
    """

    token_budget: int|None = None
    cost_budget: float|None = None
    wall_budget: float|None = None
    price_in: float|None = None
    price_out: float|None = None
    price_cache_in: float|None = None
    started: float = field(default_factory = time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_config(cls, cfg, started: float|None = None) -> 'Budget':
        """Build from a Config without making this module import it."""
        budget = cls(
            token_budget = cfg.token_budget,
            cost_budget = cfg.cost_budget,
            wall_budget = cfg.wall_budget,
            price_in = cfg.price_in,
            price_out = cfg.price_out,
            price_cache_in = cfg.price_cache_in,
        )
        if started is not None:
            budget.started = started
        return budget

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def add(self, prompt_tokens: int|None = None, completion_tokens: int|None = None,
            cached_tokens: int|None = None) -> None:
        prompt = prompt_tokens or 0
        completion = completion_tokens or 0
        cached = max(0, min(cached_tokens or 0, prompt))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached
        if self.price_in is not None and self.price_out is not None:
            cached_price = self.price_in if self.price_cache_in is None else self.price_cache_in
            self.cost += (prompt - cached) / PER_MILLION * self.price_in
            self.cost += cached / PER_MILLION * cached_price
            self.cost += completion / PER_MILLION * self.price_out
        return

    def exceeded(self) -> str|None:
        """The name of the first budget that is spent, or None."""
        if self.wall_budget is not None and self.elapsed >= self.wall_budget:
            return STOP_WALL
        if self.token_budget is not None and self.tokens >= self.token_budget:
            return STOP_TOKENS
        if self.cost_budget is not None and self.cost >= self.cost_budget:
            return STOP_COST
        return None

    def render(self) -> str:
        parts = [f'{self.tokens} tokens']
        if self.cached_tokens:
            parts.append(f'{self.cached_tokens} cached')
        if self.price_in is not None and self.price_out is not None:
            parts.append(f'${self.cost:.4f}')
        parts.append(f'{self.elapsed:.1f}s')
        return ', '.join(parts)

    def limits(self) -> str:
        """A one-line description of the configured limits, for run_start traces."""
        parts = []
        if self.token_budget is not None:
            parts.append(f'tokens={self.token_budget}')
        if self.cost_budget is not None:
            parts.append(f'cost={self.cost_budget}')
        if self.wall_budget is not None:
            parts.append(f'wall={self.wall_budget}')
        return ','.join(parts) if parts else 'none'


def cached_tokens(usage) -> int:
    """How much of a response's prompt the provider served from its cache.

    Providers disagree on where this lives: OpenAI nests it under
    ``prompt_tokens_details``, DeepSeek reports ``prompt_cache_hit_tokens``
    directly. Unknown shapes count as zero, which bills the whole prompt at the
    full rate.
    """
    if usage is None:
        return 0
    details = getattr(usage, 'prompt_tokens_details', None)
    cached = getattr(details, 'cached_tokens', None) if details is not None else None
    if cached is None:
        cached = getattr(usage, 'prompt_cache_hit_tokens', None)
    try:
        return max(0, int(cached or 0))
    except (TypeError, ValueError):
        return 0
