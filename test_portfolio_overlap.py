"""test_portfolio_overlap.py -- unit tests for portfolio_overlap.py.

Run with:  python3 -m unittest test_portfolio_overlap -v
"""
from __future__ import annotations

import unittest

from portfolio_overlap import _max_concurrent, overlap_report, simulate_capacity_constraint


def _trade(open_ts, close_ts, r=0.1, ker=None, symbol="X"):
    return {"opened_at_ts": open_ts, "closed_at_ts": close_ts, "r_multiple": r, "ker_ratio": ker, "symbol": symbol}


class SimulateCapacityConstraintTests(unittest.TestCase):
    def test_non_overlapping_trades_all_admitted_even_at_cap_one(self) -> None:
        trades = [_trade(0, 10), _trade(10, 20), _trade(20, 30)]
        admitted, rejected = simulate_capacity_constraint(trades, max_open_trades=1)
        self.assertEqual(len(admitted), 3)
        self.assertEqual(len(rejected), 0)

    def test_overlapping_trades_beyond_cap_are_rejected(self) -> None:
        # All three overlap [5,15] -- cap=1 can only ever hold the first opened.
        trades = [_trade(0, 20), _trade(5, 25), _trade(10, 30)]
        admitted, rejected = simulate_capacity_constraint(trades, max_open_trades=1)
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]["opened_at_ts"], 0)
        self.assertEqual(len(rejected), 2)

    def test_cap_of_two_admits_two_overlapping_and_rejects_the_third(self) -> None:
        trades = [_trade(0, 20), _trade(5, 25), _trade(10, 30)]
        admitted, rejected = simulate_capacity_constraint(trades, max_open_trades=2)
        self.assertEqual({t["opened_at_ts"] for t in admitted}, {0, 5})
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["opened_at_ts"], 10)

    def test_a_closed_slot_frees_up_for_a_later_open(self) -> None:
        trades = [_trade(0, 10), _trade(10, 20)]  # second opens exactly when first closes
        admitted, rejected = simulate_capacity_constraint(trades, max_open_trades=1)
        self.assertEqual(len(admitted), 2, "a trade closing AT the new open's timestamp must free its slot")
        self.assertEqual(len(rejected), 0)

    def test_simultaneous_opens_prioritized_by_higher_ker_ratio(self) -> None:
        low = _trade(0, 100, ker=0.2, symbol="LOW")
        high = _trade(0, 100, ker=0.8, symbol="HIGH")
        admitted, rejected = simulate_capacity_constraint([low, high], max_open_trades=1)
        self.assertEqual(admitted[0]["symbol"], "HIGH")
        self.assertEqual(rejected[0]["symbol"], "LOW")

    def test_missing_ker_ratio_sorts_last_not_first(self) -> None:
        no_ker = _trade(0, 100, ker=None, symbol="NOKER")
        has_ker = _trade(0, 100, ker=0.1, symbol="HASKER")  # even a low ker_ratio beats missing data
        admitted, rejected = simulate_capacity_constraint([no_ker, has_ker], max_open_trades=1)
        self.assertEqual(admitted[0]["symbol"], "HASKER")
        self.assertEqual(rejected[0]["symbol"], "NOKER")

    def test_earlier_opened_trade_keeps_its_slot_over_a_later_higher_conviction_one(self) -> None:
        """Non-simultaneous trades are never reordered by priority -- a real
        portfolio can't evict an already-open position for a later, better signal."""
        early_low_conviction = _trade(0, 100, ker=0.1, symbol="EARLY")
        later_high_conviction = _trade(1, 100, ker=0.9, symbol="LATER")
        admitted, rejected = simulate_capacity_constraint([early_low_conviction, later_high_conviction], max_open_trades=1)
        self.assertEqual(admitted[0]["symbol"], "EARLY")
        self.assertEqual(rejected[0]["symbol"], "LATER")

    def test_trades_missing_timestamps_are_silently_dropped(self) -> None:
        good = _trade(0, 10)
        bad = {"opened_at_ts": None, "closed_at_ts": 10, "r_multiple": 0.1, "ker_ratio": None}
        admitted, rejected = simulate_capacity_constraint([good, bad], max_open_trades=5)
        self.assertEqual(len(admitted) + len(rejected), 1)

    def test_invalid_cap_raises(self) -> None:
        with self.assertRaises(ValueError):
            simulate_capacity_constraint([_trade(0, 10)], max_open_trades=0)


class MaxConcurrentTests(unittest.TestCase):
    def test_no_overlap_is_one(self) -> None:
        self.assertEqual(_max_concurrent([_trade(0, 10), _trade(10, 20)]), 1)

    def test_full_overlap_of_three(self) -> None:
        self.assertEqual(_max_concurrent([_trade(0, 30), _trade(5, 25), _trade(10, 20)]), 3)

    def test_touching_boundary_does_not_count_as_overlap(self) -> None:
        self.assertEqual(_max_concurrent([_trade(0, 10), _trade(10, 20)]), 1)


class OverlapReportTests(unittest.TestCase):
    def test_reports_one_row_per_cap_with_consistent_baseline(self) -> None:
        trades = [_trade(0, 20, r=0.5), _trade(5, 25, r=-0.3), _trade(10, 30, r=0.8)]
        results = overlap_report(trades, [1, 2, 3])
        self.assertEqual([r.max_open_trades for r in results], [1, 2, 3])
        self.assertTrue(all(r.baseline_ev_R == results[0].baseline_ev_R for r in results))
        self.assertEqual(results[2].n_admitted, 3, "cap >= true peak concurrency admits everything")
        self.assertEqual(results[0].n_admitted, 1)

    def test_rejected_and_admitted_sums_account_for_every_trade(self) -> None:
        trades = [_trade(0, 20, r=0.5), _trade(5, 25, r=-0.3), _trade(10, 30, r=0.8)]
        result = overlap_report(trades, [1])[0]
        self.assertAlmostEqual(result.admitted_sum_R + result.rejected_sum_R, sum(t["r_multiple"] for t in trades), places=6)

    def test_empty_trades_returns_nan_baseline_without_crashing(self) -> None:
        results = overlap_report([], [3, 5])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.n_total, 0)


if __name__ == "__main__":
    unittest.main()
