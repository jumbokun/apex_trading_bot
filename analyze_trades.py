"""
FOMO策略交易频率分析
- 过去30天每日开单数
- 平均持仓时长
- 按小时分布可视化
"""
import os
import sys
import json
import urllib.request
import ssl
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from fomo_strategy import FOMOStrategy, FOMOStrategyConfig

def fetch_binance_klines(symbol: str, interval: str = "1m", days: int = 30) -> List[dict]:
    """从Binance获取历史K线数据"""
    binance_symbol = symbol.replace("-", "")
    if binance_symbol.startswith("1000"):
        binance_symbol = binance_symbol[4:]  # 1000PEPE -> PEPE

    ssl_ctx = ssl.create_default_context()
    all_klines = []

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"  获取 {symbol} {days}天数据...", end="", flush=True)

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


def run_backtest_detailed(symbol: str, klines: List[dict], config: FOMOStrategyConfig) -> List[Dict]:
    """回测并返回详细交易记录"""
    strategy = FOMOStrategy(config)
    trades = []
    current_trade = None

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
            current_trade = {
                "symbol": symbol,
                "entry_time": ts,
                "entry_price": signal.price,
                "side": "LONG" if signal.action == "OPEN_LONG" else "SHORT",
            }

        elif signal.action == "CLOSE" and strategy.position and current_trade:
            entry_price = strategy.position.entry_price
            exit_price = signal.price
            side = strategy.position.side

            strategy.on_trade_executed(
                symbol, "CLOSE", signal.price,
                strategy.position.quantity, ts
            )

            current_trade["exit_time"] = ts
            current_trade["exit_price"] = exit_price
            current_trade["reason"] = signal.reason

            # 计算盈亏
            if side == "LONG":
                current_trade["pnl_pct"] = (exit_price - entry_price) / entry_price * 100
            else:
                current_trade["pnl_pct"] = (entry_price - exit_price) / entry_price * 100

            # 计算持仓时长(分钟)
            duration = (ts - current_trade["entry_time"]).total_seconds() / 60
            current_trade["duration_mins"] = duration

            trades.append(current_trade)
            current_trade = None

    return trades


def visualize_hourly(hourly_counts: Dict[int, int]):
    """按小时分布可视化"""
    max_count = max(hourly_counts.values()) if hourly_counts else 1

    print("\n" + "=" * 60)
    print("开仓时间分布 (按小时UTC)")
    print("=" * 60)

    for hour in range(24):
        count = hourly_counts.get(hour, 0)
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{hour:02d}:00 | {bar:<40} {count:>3}")


def visualize_daily(daily_counts: Dict[str, int]):
    """按日期分布可视化"""
    if not daily_counts:
        return

    max_count = max(daily_counts.values())

    print("\n" + "=" * 60)
    print("每日开仓数量")
    print("=" * 60)

    for date in sorted(daily_counts.keys()):
        count = daily_counts[date]
        bar_len = int(count / max_count * 30) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{date} | {bar:<30} {count:>3}")


def main():
    print("=" * 60)
    print("FOMO策略交易频率分析 - 过去30天")
    print("=" * 60)

    # 使用与实盘相同的交易对(选取代表性的)
    SYMBOLS = [
        "BTC-USDT", "ETH-USDT", "SOL-USDT",
        "AAVE-USDT", "LINK-USDT", "UNI-USDT",
        "NEAR-USDT", "APT-USDT", "INJ-USDT",
        "ARB-USDT", "OP-USDT",
        "PEPE-USDT", "BONK-USDT", "WIF-USDT",
        "PENDLE-USDT", "CRV-USDT",
        "HYPE-USDT", "ORDI-USDT", "WLD-USDT",
        "RENDER-USDT"
    ]
    DAYS = 30

    config = FOMOStrategyConfig()

    print(f"\n策略配置: 只做多, 60周期突破, 2x成交量确认")
    print(f"回测范围: {DAYS}天")
    print(f"交易对: {len(SYMBOLS)}个")
    print("=" * 60)

    all_trades = []

    for symbol in SYMBOLS:
        klines = fetch_binance_klines(symbol, "1m", DAYS)
        if len(klines) < 1000:
            print(f"  {symbol} 数据不足，跳过")
            continue

        trades = run_backtest_detailed(symbol, klines, config)
        all_trades.extend(trades)
        print(f"  {symbol}: {len(trades)}笔交易")

    if not all_trades:
        print("没有交易记录")
        return

    # 统计
    print("\n" + "=" * 60)
    print("交易统计")
    print("=" * 60)

    # 每日统计
    daily_counts = defaultdict(int)
    hourly_counts = defaultdict(int)
    durations = []
    wins = 0

    for t in all_trades:
        date_str = t["entry_time"].strftime("%m-%d")
        daily_counts[date_str] += 1

        hour = t["entry_time"].hour
        hourly_counts[hour] += 1

        durations.append(t["duration_mins"])

        if t["pnl_pct"] > 0:
            wins += 1

    total_trades = len(all_trades)
    avg_duration = sum(durations) / len(durations) if durations else 0
    avg_daily = total_trades / DAYS

    print(f"\n总交易数: {total_trades}笔")
    print(f"平均每日: {avg_daily:.1f}笔")
    print(f"胜率: {wins/total_trades*100:.1f}%")
    print(f"\n平均持仓时长: {avg_duration:.0f}分钟 ({avg_duration/60:.1f}小时)")
    print(f"最短持仓: {min(durations):.0f}分钟")
    print(f"最长持仓: {max(durations):.0f}分钟 ({max(durations)/60:.1f}小时)")

    # 持仓时长分布
    print("\n持仓时长分布:")
    duration_buckets = {
        "< 1小时": 0,
        "1-4小时": 0,
        "4-12小时": 0,
        "12-24小时": 0,
        "> 24小时": 0
    }
    for d in durations:
        if d < 60:
            duration_buckets["< 1小时"] += 1
        elif d < 240:
            duration_buckets["1-4小时"] += 1
        elif d < 720:
            duration_buckets["4-12小时"] += 1
        elif d < 1440:
            duration_buckets["12-24小时"] += 1
        else:
            duration_buckets["> 24小时"] += 1

    for bucket, count in duration_buckets.items():
        pct = count / total_trades * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<12} | {bar:<25} {count:>3} ({pct:.1f}%)")

    # 可视化
    visualize_hourly(hourly_counts)
    visualize_daily(daily_counts)

    # 最近交易样本
    print("\n" + "=" * 60)
    print("最近10笔交易")
    print("=" * 60)
    recent = sorted(all_trades, key=lambda x: x["entry_time"], reverse=True)[:10]
    for t in recent:
        duration_str = f"{t['duration_mins']:.0f}min"
        if t['duration_mins'] >= 60:
            duration_str = f"{t['duration_mins']/60:.1f}h"
        pnl_str = f"{t['pnl_pct']:+.1f}%"
        print(f"  {t['entry_time'].strftime('%m-%d %H:%M')} {t['symbol']:<12} "
              f"{t['side']:<5} {duration_str:>8} {pnl_str:>7} | {t['reason']}")


if __name__ == "__main__":
    main()
