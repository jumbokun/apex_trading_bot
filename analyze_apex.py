"""
FOMO策略交易频率分析 - 使用Apex Exchange数据
确保成交量数据与实盘一致
"""
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from fomo_strategy import FOMOStrategy, FOMOStrategyConfig

# 尝试导入apexomni SDK
try:
    from apexomni.http_public import HttpPublic
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("apexomni SDK未安装，请运行: pip install apexomni")

# Apex API endpoint
APEX_ENDPOINT = "https://omni.apex.exchange"
apex_client = None


def init_apex_client():
    """初始化Apex客户端"""
    global apex_client
    if SDK_AVAILABLE and apex_client is None:
        apex_client = HttpPublic(APEX_ENDPOINT)
    return apex_client


def fetch_apex_klines(symbol: str, interval: str = "1", limit: int = 100) -> List[dict]:
    """
    从Apex获取K线数据 (使用SDK)

    Args:
        symbol: 交易对 (如 BTC-USDT)
        interval: K线间隔(分钟), 1/5/15/30/60/120/240/360/720/D/W/M
        limit: 获取数量 (最大1000)
    """
    client = init_apex_client()
    if not client:
        return []

    try:
        result = client.klines_v3(symbol=symbol, interval=interval, limit=limit)

        if result.get("data"):
            data = result.get("data", {})
            apex_symbol = symbol.replace("-", "")

            if isinstance(data, dict):
                kline_list = data.get(apex_symbol, [])
            else:
                kline_list = data

            klines = []
            for k in kline_list:
                klines.append({
                    "t": int(k.get("t", 0)),
                    "o": float(k.get("o", 0)),
                    "h": float(k.get("h", 0)),
                    "l": float(k.get("l", 0)),
                    "c": float(k.get("c", 0)),
                    "v": float(k.get("v", 0)),
                })
            return klines

    except Exception as e:
        print(f"获取Apex K线失败 {symbol}: {e}")

    return []


def check_apex_volume(symbols: List[str]):
    """检查Apex各币种的成交量情况"""
    print("=" * 60)
    print("Apex Exchange 成交量检查")
    print("=" * 60)

    results = []

    for symbol in symbols:
        klines = fetch_apex_klines(symbol, "1", limit=100)

        if klines:
            volumes = [k["v"] for k in klines]
            avg_vol = sum(volumes) / len(volumes) if volumes else 0
            max_vol = max(volumes) if volumes else 0
            min_vol = min(volumes) if volumes else 0
            zero_count = sum(1 for v in volumes if v == 0)

            # 检查价格是否有变化
            prices = [k["c"] for k in klines]
            price_change = (max(prices) - min(prices)) / min(prices) * 100 if min(prices) > 0 else 0

            results.append({
                "symbol": symbol,
                "avg_vol": avg_vol,
                "max_vol": max_vol,
                "zero_pct": zero_count / len(volumes) * 100,
                "price_change": price_change,
                "klines": len(klines)
            })

            status = "[OK]" if avg_vol > 0 and zero_count < len(volumes) * 0.5 else "[--]"
            print(f"{status} {symbol:<14} 平均量:{avg_vol:>12.2f}  零值:{zero_count:>3}/{len(volumes)}  波动:{price_change:.2f}%")
        else:
            print(f"[--] {symbol:<14} 无数据")

    # 筛选有效币种
    valid = [r for r in results if r["avg_vol"] > 0 and r["zero_pct"] < 50]
    print(f"\n有效币种: {len(valid)}/{len(symbols)}")

    return results


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

            if side == "LONG":
                current_trade["pnl_pct"] = (exit_price - entry_price) / entry_price * 100
            else:
                current_trade["pnl_pct"] = (entry_price - exit_price) / entry_price * 100

            duration = (ts - current_trade["entry_time"]).total_seconds() / 60
            current_trade["duration_mins"] = duration

            trades.append(current_trade)
            current_trade = None

    return trades


def visualize_hourly(hourly_counts: Dict[int, int]):
    """按小时分布可视化"""
    if not hourly_counts:
        return

    max_count = max(hourly_counts.values()) if hourly_counts else 1

    print("\n" + "=" * 60)
    print("开仓时间分布 (按小时UTC)")
    print("=" * 60)

    for hour in range(24):
        count = hourly_counts.get(hour, 0)
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{hour:02d}:00 | {bar:<40} {count:>3}")


def main():
    print("=" * 60)
    print("FOMO策略分析 - Apex Exchange数据")
    print("=" * 60)

    # 使用与实盘相同的交易对
    SYMBOLS = [
        "BTC-USDT", "ETH-USDT", "SOL-USDT",
        "AAVE-USDT", "LINK-USDT", "UNI-USDT",
        "NEAR-USDT", "APT-USDT", "INJ-USDT",
        "ARB-USDT", "OP-USDT",
        "1000PEPE-USDT", "1000BONK-USDT", "WIF-USDT",
        "PENDLE-USDT", "CRV-USDT",
        "HYPE-USDT", "ORDI-USDT", "WLD-USDT",
        "RENDER-USDT", "PENGU-USDT"
    ]

    # 第一步：检查成交量
    print("\n[步骤1] 检查Apex成交量数据")
    vol_results = check_apex_volume(SYMBOLS)

    # 筛选有效币种
    valid_symbols = [r["symbol"] for r in vol_results
                     if r["avg_vol"] > 0 and r["zero_pct"] < 50]

    if not valid_symbols:
        print("\n没有有效的交易对！")
        return

    print(f"\n[步骤2] 使用 {len(valid_symbols)} 个有效币种进行回测")
    print(f"注意: Apex API限制，只能获取最近~16小时数据 (1000条1分钟K线)")

    config = FOMOStrategyConfig()
    all_trades = []

    # 使用1分钟K线 (Apex限制最多200条，约3.3小时)
    print("  使用1分钟K线 (最近3小时)")
    for symbol in valid_symbols:
        klines = fetch_apex_klines(symbol, "1", limit=200)
        print(f"  {symbol}: {len(klines)}条K线...", end="")

        if len(klines) < 61:  # 至少需要60+1条才能计算突破
            print(" 数据不足，跳过")
            continue

        trades = run_backtest_detailed(symbol, klines, config)
        all_trades.extend(trades)
        print(f" {len(trades)}笔交易")

    if not all_trades:
        print("\n最近3小时没有交易信号 (正常情况)")
        print("原因: 60周期突破+2x成交量是严格条件，不是每小时都会触发")
        print("\n让我们分析成交量模式是否正常...")

        # 分析成交量分布
        print("\n" + "=" * 60)
        print("Apex 成交量分析 (与策略入场条件对比)")
        print("=" * 60)

        for symbol in valid_symbols[:5]:  # 只看前5个
            klines = fetch_apex_klines(symbol, "1", limit=200)
            if not klines:
                continue

            volumes = [k["v"] for k in klines]
            avg_vol = sum(volumes) / len(volumes)

            # 统计成交量超过2x均值的次数
            spikes = sum(1 for v in volumes if v >= avg_vol * 2)
            spike_pct = spikes / len(volumes) * 100

            print(f"{symbol:<14} 平均量:{avg_vol:>12.1f}  2x+放量:{spikes:>3}次 ({spike_pct:.1f}%)")

        print("\n策略入场需要: 60周期新高 + 2x成交量")
        print("如果放量比例<5%, 说明成交量条件比较难满足")
        return

    # 统计
    print("\n" + "=" * 60)
    print("交易统计 (基于Apex实际数据)")
    print("=" * 60)

    hourly_counts = defaultdict(int)
    durations = []
    wins = 0

    for t in all_trades:
        hour = t["entry_time"].hour
        hourly_counts[hour] += 1
        durations.append(t["duration_mins"])
        if t["pnl_pct"] > 0:
            wins += 1

    total_trades = len(all_trades)
    avg_duration = sum(durations) / len(durations) if durations else 0

    print(f"\n总交易数: {total_trades}笔")
    print(f"胜率: {wins/total_trades*100:.1f}%" if total_trades > 0 else "")
    print(f"平均持仓时长: {avg_duration:.0f}分钟 ({avg_duration/60:.1f}小时)")

    if durations:
        print(f"最短持仓: {min(durations):.0f}分钟")
        print(f"最长持仓: {max(durations):.0f}分钟")

    visualize_hourly(hourly_counts)

    # 显示最近交易
    print("\n" + "=" * 60)
    print("最近交易")
    print("=" * 60)
    recent = sorted(all_trades, key=lambda x: x["entry_time"], reverse=True)[:10]
    for t in recent:
        duration_str = f"{t['duration_mins']:.0f}min"
        if t['duration_mins'] >= 60:
            duration_str = f"{t['duration_mins']/60:.1f}h"
        pnl_str = f"{t['pnl_pct']:+.1f}%"
        print(f"  {t['entry_time'].strftime('%m-%d %H:%M')} {t['symbol']:<14} "
              f"{t['side']:<5} {duration_str:>8} {pnl_str:>7}")


if __name__ == "__main__":
    main()
