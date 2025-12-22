"""
参数优化测试
测试不同参数组合找到最优设置
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


def fetch_binance_klines(symbol: str, interval: str = "1m", days: int = 7) -> List[dict]:
    """从Binance获取历史K线数据"""
    binance_symbol = symbol.replace("-", "")
    ssl_ctx = ssl.create_default_context()
    all_klines = []

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

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
                        "t": k[0], "o": float(k[1]), "h": float(k[2]),
                        "l": float(k[3]), "c": float(k[4]), "v": float(k[5]),
                    })
                start_time = data[-1][0] + 1
        except:
            break
    return all_klines


def run_backtest(klines: List[dict], config: FOMOStrategyConfig, symbol: str = "TEST") -> Dict:
    """运行回测"""
    strategy = FOMOStrategy(config)

    for k in klines:
        ts = datetime.fromtimestamp(k["t"] / 1000)
        strategy.update(ts, k["o"], k["h"], k["l"], k["c"], k["v"])
        signal = strategy.generate_signal(symbol)

        if signal.action in ["OPEN_LONG", "OPEN_SHORT"]:
            strategy.on_trade_executed(symbol, signal.action, signal.price, signal.quantity, ts)
        elif signal.action == "CLOSE" and strategy.position:
            strategy.on_trade_executed(symbol, "CLOSE", signal.price, strategy.position.quantity, ts)

    status = strategy.get_status()
    return {
        "trades": status["total_trades"],
        "win_rate": status["win_rate"],
        "pnl": status["total_pnl"],
        "capital": status["capital"]
    }


def test_params():
    """测试不同参数组合"""
    print("=" * 70)
    print("FOMO策略参数优化测试")
    print("=" * 70)

    # 获取测试数据 - 更多币种验证
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "XRP-USDT", "LINK-USDT"]
    days = 14  # 测试更长时间

    print(f"\n获取测试数据 ({days}天)...")
    all_data = {}
    for sym in symbols:
        print(f"  {sym}...", end=" ")
        klines = fetch_binance_klines(sym, "1m", days)
        if len(klines) > 100:
            all_data[sym] = klines
            print(f"{len(klines)} K线")
        else:
            print("失败")

    if not all_data:
        print("无数据，退出")
        return

    # 参数组合测试 - 验证最优参数
    test_cases = [
        # 基准 (当前设置)
        {"name": "基准(当前)", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 5.0, "trail_R": 1.5},

        # 第二轮最优: 7ATR+晚2.5R
        {"name": "7ATR+晚2.5R", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 7.0, "trail_R": 2.5},

        # 在最优基础上微调
        {"name": "7ATR+晚3.0R", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 7.0, "trail_R": 3.0},
        {"name": "6ATR+晚2.5R", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 6.0, "trail_R": 2.5},
        {"name": "8ATR+晚2.5R", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 8.0, "trail_R": 2.5},

        # 调整周期
        {"name": "7+2.5+L50", "lookback": 50, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 7.0, "trail_R": 2.5},
        {"name": "7+2.5+L70", "lookback": 70, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 7.0, "trail_R": 2.5},

        # 调整止损
        {"name": "7+2.5+S1.8", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.8, "trail_k": 7.0, "trail_R": 2.5},
        {"name": "7+2.5+S2.0", "lookback": 60, "vol_ratio": 2.0, "k_stop": 2.0, "trail_k": 7.0, "trail_R": 2.5},

        # 其他备选
        {"name": "优化A", "lookback": 45, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 8.0, "trail_R": 2.0},
        {"name": "8ATR+2.0R", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 8.0, "trail_R": 2.0},
        {"name": "8ATR+2.5R", "lookback": 60, "vol_ratio": 2.0, "k_stop": 1.5, "trail_k": 8.0, "trail_R": 2.5},
    ]

    results = []

    print(f"\n测试 {len(test_cases)} 种参数组合...")
    print("-" * 70)

    for tc in test_cases:
        config = FOMOStrategyConfig(
            breakout_lookback=tc["lookback"],
            vol_ratio_entry=tc["vol_ratio"],
            k_stop_atr=tc["k_stop"],
            trail_k_atr=tc["trail_k"],
            enable_trail_after_R=tc["trail_R"],
            enable_long=True,
            enable_short=False
        )

        total_trades = 0
        total_wins = 0
        total_pnl = 0

        for sym, klines in all_data.items():
            result = run_backtest(klines, config, sym)
            total_trades += result["trades"]
            total_wins += int(result["trades"] * result["win_rate"])
            total_pnl += result["pnl"]

        win_rate = total_wins / total_trades if total_trades > 0 else 0

        results.append({
            "name": tc["name"],
            "params": tc,
            "trades": total_trades,
            "win_rate": win_rate,
            "pnl": total_pnl
        })

        status = "[+]" if total_pnl > 0 else "[-]"
        print(f"{status} {tc['name']:<15} 交易:{total_trades:>3}  胜率:{win_rate*100:>5.1f}%  盈亏:{total_pnl:>+8.2f}U")

    # 排序结果
    print("\n" + "=" * 70)
    print("按盈亏排序 TOP 5:")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: x["pnl"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        p = r["params"]
        print(f"{i+1}. {r['name']:<15} 盈亏:{r['pnl']:>+8.2f}U  胜率:{r['win_rate']*100:.1f}%")
        print(f"   参数: lookback={p['lookback']} vol={p['vol_ratio']}x stop={p['k_stop']}ATR trail={p['trail_k']}ATR @{p['trail_R']}R")

    # 最优参数
    best = sorted_results[0]
    print("\n" + "=" * 70)
    print("推荐最优参数:")
    print("=" * 70)
    bp = best["params"]
    print(f"  breakout_lookback = {bp['lookback']}")
    print(f"  vol_ratio_entry = {bp['vol_ratio']}")
    print(f"  k_stop_atr = {bp['k_stop']}")
    print(f"  trail_k_atr = {bp['trail_k']}")
    print(f"  enable_trail_after_R = {bp['trail_R']}")
    print(f"\n  预期表现: 胜率 {best['win_rate']*100:.1f}%, 盈亏 {best['pnl']:+.2f}U")


if __name__ == "__main__":
    test_params()
