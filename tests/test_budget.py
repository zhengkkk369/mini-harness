"""Budget arithmetic and the stop decision."""

import pytest

from mini_harness.budget import Budget, STOP_COST, STOP_TOKENS, STOP_WALL


def test_tokens_accumulate_across_calls():
    budget = Budget()
    budget.add(100, 20)
    budget.add(50, 10)
    assert (budget.prompt_tokens, budget.completion_tokens, budget.tokens) == (150, 30, 180)


def test_missing_usage_counts_as_zero():
    budget = Budget()
    budget.add(None, None)
    assert budget.tokens == 0


def test_without_limits_the_run_never_stops():
    budget = Budget()
    budget.add(10_000_000, 10_000_000)
    assert budget.exceeded() is None


def test_token_budget_stops_once_it_is_reached():
    budget = Budget(token_budget=100)
    budget.add(60, 39)
    assert budget.exceeded() is None
    budget.add(1, 0)
    assert budget.exceeded() == STOP_TOKENS


def test_wall_budget_is_reported_first():
    budget = Budget(token_budget=1, wall_budget=0.0)
    budget.add(10, 10)
    assert budget.exceeded() == STOP_WALL


def test_cost_is_ignored_without_prices():
    budget = Budget(cost_budget=0.01)
    budget.add(1_000_000, 1_000_000)
    assert budget.cost == 0.0
    assert budget.exceeded() is None


def test_cost_budget_stops_on_estimated_spend():
    budget = Budget(cost_budget=0.30, price_in=1.0, price_out=2.0)
    budget.add(100_000, 50_000)
    assert budget.cost == pytest.approx(0.2)
    assert budget.exceeded() is None
    budget.add(100_000, 0)
    assert budget.exceeded() == STOP_COST


def test_render_only_shows_cost_when_priced():
    assert Budget(token_budget=5).render().startswith('0 tokens')
    priced = Budget(price_in=1.0, price_out=1.0)
    priced.add(1_000_000, 0)
    assert '$1.0000' in priced.render()


def test_limits_lists_only_configured_budgets():
    assert Budget().limits() == 'none'
    assert Budget(token_budget=10).limits() == 'tokens=10'
    assert Budget(token_budget=10, wall_budget=5.0).limits() == 'tokens=10,wall=5.0'


def test_from_config_reads_every_budget_field(cfg_factory):
    cfg = cfg_factory(token_budget=1234, cost_budget=2.5, wall_budget=60.0,
                      price_in=0.5, price_out=1.5)

    budget = Budget.from_config(cfg)

    assert (budget.token_budget, budget.cost_budget, budget.wall_budget) == (1234, 2.5, 60.0)
    assert (budget.price_in, budget.price_out) == (0.5, 1.5)


def test_from_config_can_pin_the_start_time(cfg_factory):
    assert Budget.from_config(cfg_factory(), started=1.0).started == 1.0
