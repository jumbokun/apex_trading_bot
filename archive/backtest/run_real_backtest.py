"""
Run FOMO Strategy Backtest with Real BTC-USDC Data from Binance

This script:
1. Fetches real historical K-line data from Binance
2. Runs the FOMO strategy backtest
3. Compares results with different timeframes
"""
import sys
from datetime import datetime

# Import our modules
from fetch_binance_data import BinanceDataFetcher
from fomo_backtest import FOMOConfig, FOMOBacktester


def run_real_backtest(days: int = 7, interval: str = "1m"):
    """
    Run FOMO backtest with real data

    Args:
        days: Number of days to backtest
        interval: Candle interval
    """
    print("=" * 70)
    print("FOMO STRATEGY BACKTEST - REAL BTC-USDC DATA")
    print("=" * 70)
    print(f"Data Source: Binance API")
    print(f"Period: Last {days} days")
    print(f"Interval: {interval}")
    print("=" * 70)

    # Fetch real data
    print("\n[1] Fetching real market data...")
    fetcher = BinanceDataFetcher()
    timestamps, opens, highs, lows, closes, volumes = fetcher.get_historical_klines(
        symbol="BTCUSDC",
        interval=interval,
        days=days
    )

    if not timestamps:
        print("ERROR: Failed to fetch data")
        return None

    print(f"\n[2] Data loaded: {len(timestamps)} candles")
    print(f"    Range: {timestamps[0]} ~ {timestamps[-1]}")
    print(f"    Price: {min(closes):.2f} - {max(closes):.2f}")

    # Create FOMO config
    # Adjusted for real market conditions
    config = FOMOConfig(
        initial_capital=5000.0,
        risk_cash_per_campaign=50.0,

        # ATR settings
        atr_len=14,

        # Entry - more conservative for real data
        vol_ratio_entry=1.8,  # Slightly lower volume requirement
        breakout_lookback=20,
        breakout_buffer_atr=0.3,  # Larger buffer for real noise

        # Stop - adjusted for BTC volatility
        k_stop_atr=2.0,  # Wider stop for BTC
        stop_pct_min=0.005,  # 0.5% min
        stop_pct_max=0.04,   # 4% max

        # Leverage
        leverage_min=2,
        leverage_max=5,

        # Pyramid
        max_adds=3,
        add_levels_R=[1.0, 2.0, 3.0],
        add_notional_fracs=[0.6, 0.4, 0.3],
        require_stop_distance_atr=1.2,

        # Exit
        trail_k_atr=2.5,  # Tighter trailing for real data
        enable_trail_after_R=0.8,
        time_stop_bars=15,  # Slightly longer time window
        need_reach_R_in_time=0.6,

        # Global risk
        daily_loss_limit=150.0,
        max_consecutive_losses=4,
        cooldown_bars=30,

        # Execution
        commission_rate=0.0006,
        slippage=0.0005,
    )

    print(f"\n[3] Strategy Configuration:")
    print(f"    Risk per campaign:  {config.risk_cash_per_campaign}U")
    print(f"    Volume required:    {config.vol_ratio_entry}x")
    print(f"    Breakout lookback:  {config.breakout_lookback} bars")
    print(f"    Stop (ATR mult):    {config.k_stop_atr}")
    print(f"    Stop range:         {config.stop_pct_min*100:.1f}% - {config.stop_pct_max*100:.1f}%")
    print(f"    Max adds:           {config.max_adds}")
    print(f"    Trailing (ATR):     {config.trail_k_atr}")
    print(f"    Time stop:          {config.time_stop_bars} bars")

    # Run backtest
    print(f"\n[4] Running backtest...")
    backtester = FOMOBacktester(config)

    result = backtester.run_backtest(
        timestamps=timestamps,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        symbol="BTC-USDC"
    )

    # Print results
    backtester.print_result(result)

    return result


def compare_timeframes():
    """Compare strategy performance across different timeframes"""
    print("\n" + "=" * 70)
    print("TIMEFRAME COMPARISON")
    print("=" * 70)

    results = {}

    # Test different intervals
    intervals = [
        ("1m", 3),   # 3 days of 1-minute data
        ("5m", 7),   # 7 days of 5-minute data
        ("15m", 14), # 14 days of 15-minute data
    ]

    for interval, days in intervals:
        print(f"\n{'='*70}")
        print(f"Testing {interval} interval over {days} days")
        print(f"{'='*70}")

        try:
            result = run_real_backtest(days=days, interval=interval)
            if result:
                results[interval] = {
                    'days': days,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'expectancy_R': result.expectancy_R,
                    'return_pct': result.total_return_pct,
                    'max_drawdown': result.max_drawdown_pct,
                }
        except Exception as e:
            print(f"Error testing {interval}: {e}")

    # Print comparison
    if results:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Interval':<10} {'Days':<6} {'Trades':<8} {'WinRate':<10} {'PF':<8} {'Exp(R)':<10} {'Return':<10} {'MaxDD':<10}")
        print("-" * 70)

        for interval, data in results.items():
            print(f"{interval:<10} {data['days']:<6} {data['total_trades']:<8} "
                  f"{data['win_rate']:.2%}{'':4} {data['profit_factor']:<8.2f} "
                  f"{data['expectancy_R']:+.2f}R{'':4} {data['return_pct']:+.2%}{'':3} "
                  f"{data['max_drawdown']:.2%}")


if __name__ == "__main__":
    # Default: run 7 days of 1-minute data
    days = 7
    interval = "1m"

    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    if len(sys.argv) > 2:
        interval = sys.argv[2]

    result = run_real_backtest(days=days, interval=interval)

    # Optional: compare timeframes
    # compare_timeframes()
