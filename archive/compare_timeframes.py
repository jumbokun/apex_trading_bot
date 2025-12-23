"""
对比不同K线周期的回测表现
1分钟 vs 5分钟
"""
import os
import sys
import json
import urllib.request
import ssl
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
from fomo_strategy import FOMOStrategy, FOMOStrategyConfig


def fetch_binance_klines(symbol: str, interval: str = "1m", days: int = 14) -> List[dict]:
    """从Binance获取历史K线数据"""
    binance_symbol = symbol.replace("-", "")
    if binance_symbol.startswith("1000"):
        binance_symbol = binance_symbol[4:]

    ssl_ctx = ssl.create_default_context()
    all_klines = []

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"  获取 {symbol} {interval} {days}天数据...", end="", flush=True)

    while start_time < end_time:
        url = (f"https://api.binance.com/api/v3/klines?"
               f"symbol={binance_symbol}&interval={interval}&"
               f"startTime={start_time}&limit=1000")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
                data = json.loads(resp.read().decode())

                if not data:
                    break

                for k in data:
                    all_klines.append({
                        "t": k[0],
                        "o": float(k[1]),
                        "h": float(k[2]),
                        "l": float(k[3]),
                        "c": float(k[4]),
                        "v": float(k[5]),
                    })

                start_time = data[-1][0] + 1
                print(".", end="", flush=True)

        except Exception as e:
            print(f" 失败: {e}")
            break

    print(f" {len(all_klines)}条")
    return all_klines


def run_backtest(symbol: str, klines: List[dict], config: FOMOStrategyConfig) -> Dict:
    """运行回测"""
    strategy = FOMOStrategy(config)
    trades = []

    for k in klines:
        ts = datetime.fromtimestamp(k["t"] / 1000)

        strategy.update(
            timestamp=ts,
            open_=k["o"],
            high=k["h"],
            low=k["l"],
            close=k["c"],
            volume=k["v"]
        )

        signal = strategy.generate_signal(symbol)

        if signal.action in ["OPEN_LONG", "OPEN_SHORT"]:
            strategy.on_trade_executed(
                symbol, signal.action, signal.price,
                signal.quantity, ts
            )
            trades.append({
                "time": ts,
                "action": signal.action,
                "price": signal.price,
            })

        elif signal.action == "CLOSE" and strategy.position:
            entry_price = strategy.position.entry_price
            side = strategy.position.side

            strategy.on_trade_executed(
                symbol, "CLOSE", signal.price,
                strategy.position.quantity, ts
            )
            trades.append({
                "time": ts,
                "action": f"CLOSE_{side}",
                "price": signal.price,
                "entry_price": entry_price,
            })

    status = strategy.get_status()

    return {
        "symbol": symbol,
        "total_trades": status["total_trades"],
        "winning_trades": status["winning_trades"],
        "win_rate": status["win_rate"],
        "total_pnl": status["total_pnl"],
        "trades": trades
    }


def main():
    print("=" * 70)
    print("K线周期对比回测: 1分钟 vs 5分钟")
    print("=" * 70)

    # 测试币种
    SYMBOLS = [
        "BTC-USDT", "ETH-USDT", "SOL-USDT",
        "LINK-USDT", "AAVE-USDT", "ARB-USDT",
    ]
    DAYS = 14

    # 使用当前策略配置
    config = FOMOStrategyConfig()

    print(f"\n策略配置:")
    print(f"  突破周期: {config.breakout_lookback}")
    print(f"  成交量要求: {config.vol_ratio_entry}x")
    print(f"  止损ATR: {config.k_stop_atr}")
    print(f"  追踪ATR: {config.trail_k_atr}")
    print(f"  追踪启用: {config.enable_trail_after_R}R")
    print(f"  仓位范围: {config.notional_min}-{config.notional_max}U")

    print(f"\n回测范围: {DAYS}天")
    print(f"测试币种: {len(SYMBOLS)}个")

    # ========== 1分钟回测 ==========
    print("\n" + "=" * 70)
    print("【1分钟K线回测】")
    print("=" * 70)

    results_1m = []
    total_pnl_1m = 0
    total_trades_1m = 0
    total_wins_1m = 0

    for symbol in SYMBOLS:
        klines = fetch_binance_klines(symbol, "1m", DAYS)
        if len(klines) < 100:
            print(f"  {symbol} 数据不足，跳过")
            continue

        result = run_backtest(symbol, klines, config)
        results_1m.append(result)

        total_pnl_1m += result["total_pnl"]
        total_trades_1m += result["total_trades"]
        total_wins_1m += result["winning_trades"]

        print(f"  {symbol}: {result['total_trades']}笔, "
              f"胜率{result['win_rate']*100:.1f}%, "
              f"盈亏{result['total_pnl']:+.2f}U")

    # ========== 5分钟回测 ==========
    print("\n" + "=" * 70)
    print("【5分钟K线回测】")
    print("=" * 70)

    results_5m = []
    total_pnl_5m = 0
    total_trades_5m = 0
    total_wins_5m = 0

    for symbol in SYMBOLS:
        klines = fetch_binance_klines(symbol, "5m", DAYS)
        if len(klines) < 100:
            print(f"  {symbol} 数据不足，跳过")
            continue

        result = run_backtest(symbol, klines, config)
        results_5m.append(result)

        total_pnl_5m += result["total_pnl"]
        total_trades_5m += result["total_trades"]
        total_wins_5m += result["winning_trades"]

        print(f"  {symbol}: {result['total_trades']}笔, "
              f"胜率{result['win_rate']*100:.1f}%, "
              f"盈亏{result['total_pnl']:+.2f}U")

    # ========== 对比汇总 ==========
    print("\n" + "=" * 70)
    print("【对比汇总】")
    print("=" * 70)

    print(f"\n{'指标':<20} {'1分钟':<20} {'5分钟':<20}")
    print("-" * 60)
    print(f"{'总交易数':<20} {total_trades_1m:<20} {total_trades_5m:<20}")

    win_rate_1m = total_wins_1m / total_trades_1m * 100 if total_trades_1m > 0 else 0
    win_rate_5m = total_wins_5m / total_trades_5m * 100 if total_trades_5m > 0 else 0
    print(f"{'胜率':<20} {win_rate_1m:.1f}%{'':<15} {win_rate_5m:.1f}%")

    print(f"{'总盈亏':<20} {total_pnl_1m:+.2f}U{'':<12} {total_pnl_5m:+.2f}U")

    avg_pnl_per_trade_1m = total_pnl_1m / total_trades_1m if total_trades_1m > 0 else 0
    avg_pnl_per_trade_5m = total_pnl_5m / total_trades_5m if total_trades_5m > 0 else 0
    print(f"{'平均每笔盈亏':<20} {avg_pnl_per_trade_1m:+.2f}U{'':<12} {avg_pnl_per_trade_5m:+.2f}U")

    trades_per_day_1m = total_trades_1m / DAYS
    trades_per_day_5m = total_trades_5m / DAYS
    print(f"{'每日交易数':<20} {trades_per_day_1m:.1f}{'':<17} {trades_per_day_5m:.1f}")

    # 推荐
    print("\n" + "=" * 70)
    print("【推荐】")
    print("=" * 70)

    if total_pnl_5m > total_pnl_1m:
        print(f"5分钟K线表现更好 (盈利 {total_pnl_5m:+.2f}U vs {total_pnl_1m:+.2f}U)")
        print("优点: 更少噪音，信号更稳定，交易频率更低")
    else:
        print(f"1分钟K线表现更好 (盈利 {total_pnl_1m:+.2f}U vs {total_pnl_5m:+.2f}U)")
        print("优点: 更快响应，捕捉更多机会")

    if win_rate_5m > win_rate_1m:
        print(f"5分钟胜率更高 ({win_rate_5m:.1f}% vs {win_rate_1m:.1f}%)")
    else:
        print(f"1分钟胜率更高 ({win_rate_1m:.1f}% vs {win_rate_5m:.1f}%)")


if __name__ == "__main__":
    main()
