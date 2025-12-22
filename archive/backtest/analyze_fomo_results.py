"""
Analyze FOMO Strategy Backtest Results - Identify Failure Patterns

This script analyzes why the FOMO strategy is underperforming and suggests improvements.
"""
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict

from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import FOMOConfig, FOMOBacktester, BacktestResult, Trade


def analyze_exit_reasons(trades: List[Trade]) -> Dict[str, Dict]:
    """Analyze distribution of exit reasons"""
    exit_stats = defaultdict(lambda: {"count": 0, "total_pnl": 0, "wins": 0, "losses": 0})

    for trade in trades:
        # Categorize exit reason
        reason = trade.exit_reason
        if "Time Stop" in reason:
            category = "Time Stop"
        elif "Trailing Stop" in reason:
            category = "Trailing Stop"
        elif "Stop Loss" in reason:
            category = "Stop Loss"
        elif "End of Backtest" in reason:
            category = "End of Backtest"
        else:
            category = "Other"

        exit_stats[category]["count"] += 1
        exit_stats[category]["total_pnl"] += trade.pnl
        if trade.pnl > 0:
            exit_stats[category]["wins"] += 1
        else:
            exit_stats[category]["losses"] += 1

    return dict(exit_stats)


def analyze_time_in_trade(trades: List[Trade]) -> Dict:
    """Analyze how long trades last"""
    durations = []
    for trade in trades:
        if trade.entry_time and trade.exit_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 60  # minutes
            durations.append(duration)

    if not durations:
        return {}

    import numpy as np
    return {
        "avg_duration_min": np.mean(durations),
        "median_duration_min": np.median(durations),
        "min_duration_min": np.min(durations),
        "max_duration_min": np.max(durations),
        "under_5min": sum(1 for d in durations if d < 5),
        "under_12min": sum(1 for d in durations if d < 12),
        "over_30min": sum(1 for d in durations if d > 30),
    }


def analyze_r_multiples(trades: List[Trade]) -> Dict:
    """Analyze R-multiple distribution"""
    r_multiples = [t.r_multiple for t in trades]

    if not r_multiples:
        return {}

    import numpy as np

    # Categorize R-multiples
    big_wins = [r for r in r_multiples if r >= 2.0]     # >= +2R
    small_wins = [r for r in r_multiples if 0 < r < 2.0]  # +0 to +2R
    scratches = [r for r in r_multiples if -0.5 <= r <= 0]  # -0.5R to 0R
    small_losses = [r for r in r_multiples if -1.0 <= r < -0.5]  # -1R to -0.5R
    full_losses = [r for r in r_multiples if r < -1.0]  # < -1R (worse than expected)

    return {
        "avg_r": np.mean(r_multiples),
        "median_r": np.median(r_multiples),
        "best_r": np.max(r_multiples),
        "worst_r": np.min(r_multiples),
        "big_wins_count": len(big_wins),
        "small_wins_count": len(small_wins),
        "scratches_count": len(scratches),
        "small_losses_count": len(small_losses),
        "full_losses_count": len(full_losses),
        "big_wins_pct": len(big_wins) / len(r_multiples) * 100 if r_multiples else 0,
        "full_losses_pct": len(full_losses) / len(r_multiples) * 100 if r_multiples else 0,
    }


def analyze_adds(trades: List[Trade]) -> Dict:
    """Analyze pyramid adding behavior"""
    add_counts = [t.add_count for t in trades]

    if not add_counts:
        return {}

    import numpy as np

    # Trades with adds vs no adds
    with_adds = [t for t in trades if t.add_count > 0]
    without_adds = [t for t in trades if t.add_count == 0]

    return {
        "avg_adds": np.mean(add_counts),
        "trades_with_adds": len(with_adds),
        "trades_without_adds": len(without_adds),
        "add_rate": len(with_adds) / len(trades) * 100 if trades else 0,
        "avg_pnl_with_adds": np.mean([t.pnl for t in with_adds]) if with_adds else 0,
        "avg_pnl_without_adds": np.mean([t.pnl for t in without_adds]) if without_adds else 0,
        "win_rate_with_adds": sum(1 for t in with_adds if t.pnl > 0) / len(with_adds) * 100 if with_adds else 0,
        "win_rate_without_adds": sum(1 for t in without_adds if t.pnl > 0) / len(without_adds) * 100 if without_adds else 0,
    }


def run_analysis():
    """Run comprehensive analysis on multi-pair backtest"""

    print("=" * 70)
    print("FOMO STRATEGY FAILURE ANALYSIS")
    print("=" * 70)

    # Fetch data for hot pairs
    fetcher = MultiPairFetcher()
    print("\n[1] Fetching hot pairs...")
    hot_pairs = fetcher.get_hot_pairs(top_n=10)
    pairs = [p.get('symbol') for p in hot_pairs]

    print(f"\n[2] Fetching 3 days of 1m data for {len(pairs)} pairs...")
    all_data = fetcher.fetch_all_pairs_data(pairs, "1m", 3)

    if not all_data:
        print("ERROR: No data fetched")
        return

    # Run backtest with current config
    config = FOMOConfig(
        initial_capital=5000.0,
        risk_cash_per_campaign=50.0,
        atr_len=14,
        vol_ratio_entry=1.8,
        breakout_lookback=20,
        breakout_buffer_atr=0.25,
        k_stop_atr=1.8,
        stop_pct_min=0.008,
        stop_pct_max=0.05,
        leverage_min=2,
        leverage_max=5,
        max_adds=3,
        add_levels_R=[1.0, 2.0, 3.0],
        add_notional_fracs=[0.6, 0.4, 0.3],
        require_stop_distance_atr=1.0,
        trail_k_atr=2.0,
        enable_trail_after_R=0.8,
        time_stop_bars=20,
        need_reach_R_in_time=0.5,
        daily_loss_limit=150.0,
        max_consecutive_losses=5,
        cooldown_bars=20,
        commission_rate=0.0006,
        slippage=0.001,
    )

    print(f"\n[3] Running backtests and collecting trade data...")

    all_trades = []
    all_results = {}

    for symbol, data in all_data.items():
        timestamps, opens, highs, lows, closes, volumes = data

        if len(timestamps) < 100:
            continue

        backtester = FOMOBacktester(config)

        try:
            # Suppress individual trade output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            result = backtester.run_backtest(
                timestamps=timestamps,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                symbol=symbol
            )

            sys.stdout = old_stdout

            all_results[symbol] = result
            all_trades.extend(result.trades)

        except Exception as e:
            sys.stdout = old_stdout
            print(f"  {symbol}: Error - {e}")

    print(f"\n  Analyzed {len(all_results)} pairs, {len(all_trades)} total trades")

    # ========== ANALYSIS ==========

    print("\n" + "=" * 70)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 70)

    # 1. Exit Reason Analysis
    print("\n[A] EXIT REASON BREAKDOWN")
    print("-" * 50)
    exit_stats = analyze_exit_reasons(all_trades)

    total_trades = len(all_trades)
    for reason, stats in sorted(exit_stats.items(), key=lambda x: -x[1]["count"]):
        pct = stats["count"] / total_trades * 100 if total_trades else 0
        avg_pnl = stats["total_pnl"] / stats["count"] if stats["count"] else 0
        win_rate = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
        print(f"  {reason:<20}: {stats['count']:>4} trades ({pct:>5.1f}%) | "
              f"Avg PnL: {avg_pnl:>+8.2f} | Win Rate: {win_rate:>5.1f}%")

    # 2. Time Analysis
    print("\n[B] TRADE DURATION ANALYSIS")
    print("-" * 50)
    time_stats = analyze_time_in_trade(all_trades)
    if time_stats:
        print(f"  Avg Duration:    {time_stats['avg_duration_min']:>8.1f} minutes")
        print(f"  Median Duration: {time_stats['median_duration_min']:>8.1f} minutes")
        print(f"  Min Duration:    {time_stats['min_duration_min']:>8.1f} minutes")
        print(f"  Max Duration:    {time_stats['max_duration_min']:>8.1f} minutes")
        print(f"  Under 5 min:     {time_stats['under_5min']:>8} trades ({time_stats['under_5min']/total_trades*100:.1f}%)")
        print(f"  Under 12 min:    {time_stats['under_12min']:>8} trades ({time_stats['under_12min']/total_trades*100:.1f}%)")
        print(f"  Over 30 min:     {time_stats['over_30min']:>8} trades ({time_stats['over_30min']/total_trades*100:.1f}%)")

    # 3. R-Multiple Analysis
    print("\n[C] R-MULTIPLE DISTRIBUTION")
    print("-" * 50)
    r_stats = analyze_r_multiples(all_trades)
    if r_stats:
        print(f"  Average R:        {r_stats['avg_r']:>+8.2f}R")
        print(f"  Median R:         {r_stats['median_r']:>+8.2f}R")
        print(f"  Best Trade:       {r_stats['best_r']:>+8.2f}R")
        print(f"  Worst Trade:      {r_stats['worst_r']:>+8.2f}R")
        print(f"")
        print(f"  Big Wins (>=+2R):     {r_stats['big_wins_count']:>4} ({r_stats['big_wins_pct']:>5.1f}%)")
        print(f"  Small Wins (0 to +2R):{r_stats['small_wins_count']:>4}")
        print(f"  Scratches (-0.5 to 0):{r_stats['scratches_count']:>4}")
        print(f"  Small Losses:         {r_stats['small_losses_count']:>4}")
        print(f"  Full Losses (<-1R):   {r_stats['full_losses_count']:>4} ({r_stats['full_losses_pct']:>5.1f}%)")

    # 4. Pyramid Adding Analysis
    print("\n[D] PYRAMID ADDING ANALYSIS")
    print("-" * 50)
    add_stats = analyze_adds(all_trades)
    if add_stats:
        print(f"  Avg Adds/Trade:      {add_stats['avg_adds']:>8.2f}")
        print(f"  Trades with adds:    {add_stats['trades_with_adds']:>8} ({add_stats['add_rate']:.1f}%)")
        print(f"  Trades without adds: {add_stats['trades_without_adds']:>8}")
        print(f"")
        print(f"  With Adds:")
        print(f"    Avg PnL:           {add_stats['avg_pnl_with_adds']:>+8.2f}")
        print(f"    Win Rate:          {add_stats['win_rate_with_adds']:>8.1f}%")
        print(f"  Without Adds:")
        print(f"    Avg PnL:           {add_stats['avg_pnl_without_adds']:>+8.2f}")
        print(f"    Win Rate:          {add_stats['win_rate_without_adds']:>8.1f}%")

    # ========== DIAGNOSIS ==========

    print("\n" + "=" * 70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 70)

    issues = []
    recommendations = []

    # Check time stop issue
    time_stop_pct = exit_stats.get("Time Stop", {}).get("count", 0) / total_trades * 100 if total_trades else 0
    if time_stop_pct > 50:
        issues.append(f"Time Stop triggered on {time_stop_pct:.1f}% of trades - too aggressive")
        recommendations.append("Increase time_stop_bars from 20 to 40-60 bars")
        recommendations.append("Lower need_reach_R_in_time from 0.5 to 0.3")

    # Check stop loss issue
    stop_loss_pct = exit_stats.get("Stop Loss", {}).get("count", 0) / total_trades * 100 if total_trades else 0
    if stop_loss_pct > 40:
        issues.append(f"Stop Loss hit on {stop_loss_pct:.1f}% of trades - stops too tight")
        recommendations.append("Increase k_stop_atr from 1.8 to 2.5-3.0")
        recommendations.append("Raise stop_pct_min from 0.8% to 1.2%")

    # Check for lack of big wins
    if r_stats and r_stats['big_wins_pct'] < 5:
        issues.append(f"Only {r_stats['big_wins_pct']:.1f}% of trades reached +2R - not letting winners run")
        recommendations.append("Widen trailing stop (increase trail_k_atr from 2.0 to 3.0)")
        recommendations.append("Enable trailing later (increase enable_trail_after_R from 0.8 to 1.5)")

    # Check for too many full losses
    if r_stats and r_stats['full_losses_pct'] > 30:
        issues.append(f"{r_stats['full_losses_pct']:.1f}% of trades lost more than 1R")
        recommendations.append("Ensure risk cap is working properly")
        recommendations.append("Review slippage settings")

    # Check for false breakouts
    if time_stats and time_stats['under_5min'] / total_trades > 0.3:
        issues.append(f"{time_stats['under_5min']/total_trades*100:.1f}% of trades exit within 5 min - many false breakouts")
        recommendations.append("Require higher volume ratio (3.0x instead of 1.8x)")
        recommendations.append("Increase breakout lookback from 20 to 30-40 bars")
        recommendations.append("Add trend filter (only trade in direction of 50-period EMA)")

    # Check adding effectiveness
    if add_stats and add_stats['win_rate_with_adds'] < add_stats['win_rate_without_adds']:
        issues.append("Trades with adds have LOWER win rate - adds may be hurting")
        recommendations.append("Increase add R-levels (1.5R, 2.5R, 3.5R instead of 1R, 2R, 3R)")
        recommendations.append("Require stronger confirmation for adds (volume 2.0x instead of 1.3x)")

    # Market condition
    total_pnl = sum(r.total_return for r in all_results.values())
    profitable_count = sum(1 for r in all_results.values() if r.total_return > 0)

    if profitable_count == 0:
        issues.append("0 profitable pairs - market may be sideways/consolidating")
        recommendations.append("Add market regime filter (ADX > 20 for trending)")
        recommendations.append("Consider longer timeframe (5m or 15m instead of 1m)")
        recommendations.append("Reduce trade frequency - only take highest conviction setups")

    print("\n[ISSUES IDENTIFIED]")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n[RECOMMENDATIONS]")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    # ========== SUGGESTED NEW CONFIG ==========

    print("\n" + "=" * 70)
    print("SUGGESTED PARAMETER ADJUSTMENTS")
    print("=" * 70)
    print("""
CONSERVATIVE TREND-FOLLOWING CONFIG:

  # Entry - Be more selective
  vol_ratio_entry: 3.0        # Was 1.8 - require stronger volume
  breakout_lookback: 40       # Was 20 - longer lookback for cleaner breakouts
  breakout_buffer_atr: 0.4    # Was 0.25 - more buffer to filter noise

  # Stop Loss - Give more room
  k_stop_atr: 2.5             # Was 1.8 - wider stops
  stop_pct_min: 0.012         # Was 0.008 - 1.2% minimum stop
  stop_pct_max: 0.06          # Keep at 6%

  # Time Management - Be patient
  time_stop_bars: 60          # Was 20 - 1 hour instead of 20 min
  need_reach_R_in_time: 0.3   # Was 0.5 - lower threshold

  # Trailing Stop - Let winners run
  trail_k_atr: 3.0            # Was 2.0 - wider trailing
  enable_trail_after_R: 1.5   # Was 0.8 - wait longer to enable

  # Adds - Higher bar for adding
  add_levels_R: [1.5, 2.5, 3.5]  # Was [1.0, 2.0, 3.0]
  vol_ratio_add: 2.0          # Was 1.3 - stronger confirmation

  # Consider adding:
  - Trend filter (only long when price > EMA50)
  - ADX filter (only trade when ADX > 20)
  - Use 5m or 15m timeframe instead of 1m
""")

    return all_results, all_trades


if __name__ == "__main__":
    run_analysis()
