from risk.nav_allocator import apply_nav_allocator, load_budgets, BucketBudgets


def test_pass_through_when_disabled() -> None:
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    buckets = {"AAPL": "ts_core", "MSFT": "ts_core"}
    budgets = BucketBudgets(enabled=False, base_nav_frac={}, min_nav_frac={}, max_nav_frac={})
    adjusted = apply_nav_allocator(weights, buckets, budgets, nav=100000.0)
    assert adjusted == weights


def test_allocation_respects_budgets() -> None:
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    buckets = {"AAPL": "ts_core", "MSFT": "ts_core"}
    budgets = BucketBudgets(
        enabled=True,
        base_nav_frac={"ts_core": 0.3},
        min_nav_frac={"ts_core": 0.2},
        max_nav_frac={"ts_core": 0.4},
    )
    adjusted = apply_nav_allocator(weights, buckets, budgets, nav=100000.0)
    # Base 30% NAV budget on 100k => 30k allocated to ts_core.
    # AAPL gets 60% of that, MSFT 40%.
    assert round(adjusted["AAPL"] * 100000.0, 2) == 18000.0
    assert round(adjusted["MSFT"] * 100000.0, 2) == 12000.0
