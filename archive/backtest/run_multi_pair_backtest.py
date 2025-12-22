"""
Run FOMO Strategy Backtest with Multiple Trading Pairs

This simulates scanning multiple coins for hot momentum plays,
which is more realistic for a FOMO strategy
"""
import sys
from datetime import datetime
from typing import Dict, List, Tuple

from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import FOMOConfig, FOMOBacktester, BacktestResult


def run_multi_pair_backtest(
    pairs: List[str] = None,
    days: int = 3,
    interval: str = "1m"
) -> Dict[str, BacktestResult]:
    """
    Run FOMO backtest on multiple pairs

    Args:
        pairs: List of trading pairs (None = use hot pairs)
        days: Number of days
        interval: Candle interval

    Returns:
        Dict of symbol -> BacktestResult
    """
    print("=" * 70)
    print("FOMO STRATEGY - MULTI-PAIR BACKTEST")
    print("=" * 70)

    fetcher = MultiPairFetcher()

    # Get hot pairs if not specified
    if pairs is None:
        print("\n[1] Scanning for hot pairs...")
        hot_pairs = fetcher.get_hot_pairs(top_n=10)
        pairs = [p.get('symbol') for p in hot_pairs]

        print(f"\nSelected {len(pairs)} pairs based on 24h activity:")
        for i, p in enumerate(hot_pairs):
            print(f"  {i+1}. {p.get('symbol'):<12} Change: {p.get('priceChangeFloat', 0):+.2f}%  "
                  f"Volatility: {p.get('volatility', 0):.2f}%")

    # Fetch data for all pairs
    print(f"\n[2] Fetching {interval} data for {len(pairs)} pairs ({days} days)...")
    all_data = fetcher.fetch_all_pairs_data(pairs, interval, days)

    if not all_data:
        print("ERROR: No data fetched")
        return {}

    # Create config
    config = FOMOConfig(
        initial_capital=5000.0,
        risk_cash_per_campaign=50.0,

        atr_len=14,

        # Entry
        vol_ratio_entry=1.8,
        breakout_lookback=20,
        breakout_buffer_atr=0.25,

        # Stop - adjusted for crypto volatility
        k_stop_atr=1.8,
        stop_pct_min=0.008,  # 0.8% min for alts
        stop_pct_max=0.05,   # 5% max

        # Leverage
        leverage_min=2,
        leverage_max=5,

        # Pyramid
        max_adds=3,
        add_levels_R=[1.0, 2.0, 3.0],
        add_notional_fracs=[0.6, 0.4, 0.3],
        require_stop_distance_atr=1.0,

        # Exit
        trail_k_atr=2.0,
        enable_trail_after_R=0.8,
        time_stop_bars=20,  # 20 minutes for 1m
        need_reach_R_in_time=0.5,

        # Global risk
        daily_loss_limit=150.0,
        max_consecutive_losses=5,
        cooldown_bars=20,

        # Execution
        commission_rate=0.0006,
        slippage=0.001,  # Higher slippage for alts
    )

    print(f"\n[3] Strategy Config:")
    print(f"    Risk/Campaign:   {config.risk_cash_per_campaign}U")
    print(f"    Stop Range:      {config.stop_pct_min*100:.1f}% - {config.stop_pct_max*100:.1f}%")
    print(f"    Time Stop:       {config.time_stop_bars} bars")
    print(f"    Vol Required:    {config.vol_ratio_entry}x")

    # Run backtest on each pair
    print(f"\n[4] Running backtests...")
    print("=" * 70)

    results = {}
    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for symbol, data in all_data.items():
        timestamps, opens, highs, lows, closes, volumes = data

        if len(timestamps) < 100:
            print(f"  {symbol}: Skipped (not enough data)")
            continue

        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print(f"{'='*70}")

        backtester = FOMOBacktester(config)

        try:
            result = backtester.run_backtest(
                timestamps=timestamps,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                symbol=symbol
            )

            results[symbol] = result
            total_pnl += result.total_return
            total_trades += result.total_trades
            total_wins += result.winning_trades

            # Brief summary
            print(f"\n  Result: {result.total_return:+.2f} USDC ({result.total_return_pct:+.2%})")
            print(f"  Trades: {result.total_trades} | Win Rate: {result.win_rate:.2%}")
            print(f"  Expectancy: {result.expectancy_R:+.2f}R | Max DD: {result.max_drawdown_pct:.2%}")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print(f"\n{'Symbol':<12} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Return':>10} {'Exp(R)':>10} {'MaxDD':>10}")
    print("-" * 70)

    for symbol, result in sorted(results.items(), key=lambda x: x[1].total_return, reverse=True):
        print(f"{symbol:<12} {result.total_trades:>8} {result.win_rate:>7.1%} "
              f"{result.total_return:>+12.2f} {result.total_return_pct:>+9.2%} "
              f"{result.expectancy_R:>+9.2f}R {result.max_drawdown_pct:>9.2%}")

    print("-" * 70)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    print(f"{'TOTAL':<12} {total_trades:>8} {overall_wr:>7.1%} {total_pnl:>+12.2f}")

    # Best and worst
    if results:
        best = max(results.items(), key=lambda x: x[1].total_return)
        worst = min(results.items(), key=lambda x: x[1].total_return)
        print(f"\nBest:  {best[0]} with {best[1].total_return:+.2f} USDC ({best[1].total_return_pct:+.2%})")
        print(f"Worst: {worst[0]} with {worst[1].total_return:+.2f} USDC ({worst[1].total_return_pct:+.2%})")

        # Profitable pairs
        profitable = [s for s, r in results.items() if r.total_return > 0]
        print(f"\nProfitable pairs: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.0f}%)")

    return results


if __name__ == "__main__":
    days = 3
    interval = "1m"

    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    if len(sys.argv) > 2:
        interval = sys.argv[2]

    results = run_multi_pair_backtest(days=days, interval=interval)
