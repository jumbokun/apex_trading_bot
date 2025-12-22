"""
FOMO策略回测 - 支持双向交易
使用Binance历史数据进行回测
"""
import os
import sys
import json
import urllib.request
import ssl
from datetime import datetime, timedelta
from typing import List, Dict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from fomo_strategy import FOMOStrategy, FOMOStrategyConfig, Signal


def fetch_binance_klines(symbol: str, interval: str = "1m",
                         days: int = 7, limit: int = 1000) -> List[dict]:
    """从Binance获取历史K线数据"""
    # 转换交易对格式: BTC-USDT -> BTCUSDT
    binance_symbol = symbol.replace("-", "")

    ssl_ctx = ssl.create_default_context()
    all_klines = []

    # 计算需要获取的时间范围
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"  获取 {symbol} {days}天数据...")

    while start_time < end_time:
        url = (f"https://api.binance.com/api/v3/klines?"
               f"symbol={binance_symbol}&interval={interval}&"
               f"startTime={start_time}&limit={limit}")

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

                # 更新起始时间
                start_time = data[-1][0] + 1

        except Exception as e:
            print(f"    获取数据失败: {e}")
            break

    print(f"    获取 {len(all_klines)} 根K线")
    return all_klines


def run_backtest(symbol: str, klines: List[dict], config: FOMOStrategyConfig) -> Dict:
    """运行单个交易对的回测"""
    strategy = FOMOStrategy(config)

    trades = []

    for k in klines:
        ts = datetime.fromtimestamp(k["t"] / 1000)

        # 更新数据
        strategy.update(
            timestamp=ts,
            open_=k["o"],
            high=k["h"],
            low=k["l"],
            close=k["c"],
            volume=k["v"]
        )

        # 生成信号
        signal = strategy.generate_signal(symbol)

        # 处理信号
        if signal.action in ["OPEN_LONG", "OPEN_SHORT"]:
            strategy.on_trade_executed(
                symbol, signal.action, signal.price,
                signal.quantity, ts
            )
            trades.append({
                "time": ts,
                "action": signal.action,
                "price": signal.price,
                "quantity": signal.quantity,
                "reason": signal.reason
            })

        elif signal.action == "CLOSE" and strategy.position:
            # 记录平仓前的信息
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
                "reason": signal.reason
            })

    status = strategy.get_status()

    return {
        "symbol": symbol,
        "total_trades": status["total_trades"],
        "winning_trades": status["winning_trades"],
        "win_rate": status["win_rate"],
        "total_pnl": status["total_pnl"],
        "final_capital": status["capital"],
        "trades": trades
    }


def main():
    """主函数"""
    print("=" * 60)
    print("FOMO策略回测 - 双向交易版")
    print("=" * 60)

    # 回测配置
    SYMBOLS = [
        "BTC-USDT", "ETH-USDT", "SOL-USDT",
        "1000PEPE-USDT", "DOGE-USDT", "XRP-USDT"
    ]
    DAYS = 7  # 回测天数

    config = FOMOStrategyConfig()

    print(f"\n策略配置:")
    print(f"  突破周期: {config.breakout_lookback}")
    print(f"  成交量要求: {config.vol_ratio_entry}x")
    print(f"  止损ATR倍数: {config.k_stop_atr}")
    print(f"  追踪ATR倍数: {config.trail_k_atr}")
    print(f"  启用追踪: {config.enable_trail_after_R}R")
    print(f"  单笔风险: {config.risk_per_trade}U")

    print(f"\n回测范围: {DAYS}天")
    print(f"交易对: {SYMBOLS}")
    print("=" * 60)

    # 获取数据并回测
    all_results = []
    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for symbol in SYMBOLS:
        print(f"\n[{symbol}]")

        # 获取数据
        klines = fetch_binance_klines(symbol, "1m", DAYS)
        if len(klines) < 100:
            print(f"  数据不足，跳过")
            continue

        # 运行回测
        result = run_backtest(symbol, klines, config)
        all_results.append(result)

        total_pnl += result["total_pnl"]
        total_trades += result["total_trades"]
        total_wins += result["winning_trades"]

        print(f"  交易次数: {result['total_trades']}")
        print(f"  胜率: {result['win_rate']*100:.1f}%")
        print(f"  盈亏: {result['total_pnl']:+.2f}U")

        # 显示最近交易
        if result["trades"]:
            print(f"  最近交易:")
            for t in result["trades"][-5:]:
                print(f"    {t['time'].strftime('%m-%d %H:%M')} {t['action']:<12} @ {t['price']:.2f} - {t['reason']}")

    # 汇总
    print("\n" + "=" * 60)
    print("回测汇总")
    print("=" * 60)
    print(f"总交易次数: {total_trades}")
    print(f"总胜率: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "总胜率: N/A")
    print(f"总盈亏: {total_pnl:+.2f}U")
    print(f"初始资金: {config.initial_capital}U")
    print(f"收益率: {total_pnl/config.initial_capital*100:+.2f}%")

    # 按盈亏排序
    if all_results:
        print(f"\n各品种表现:")
        sorted_results = sorted(all_results, key=lambda x: x["total_pnl"], reverse=True)
        for r in sorted_results:
            status = "[+]" if r["total_pnl"] > 0 else "[-]"
            print(f"  {status} {r['symbol']:<15} 交易:{r['total_trades']:>3}  "
                  f"胜率:{r['win_rate']*100:>5.1f}%  盈亏:{r['total_pnl']:>+8.2f}U")

    # 分析多空表现
    print("\n" + "=" * 60)
    print("多空分析")
    print("=" * 60)
    long_count = 0
    long_wins = 0
    short_count = 0
    short_wins = 0

    for r in all_results:
        for t in r["trades"]:
            if t["action"] == "OPEN_LONG":
                pass  # 开仓不计入
            elif t["action"] == "CLOSE_LONG":
                long_count += 1
                if "追踪" in t.get("reason", ""):
                    long_wins += 1
            elif t["action"] == "OPEN_SHORT":
                pass
            elif t["action"] == "CLOSE_SHORT":
                short_count += 1
                if "追踪" in t.get("reason", ""):
                    short_wins += 1

    if long_count > 0:
        print(f"多头: {long_count}笔, 追踪止盈:{long_wins}笔, 止盈率:{long_wins/long_count*100:.1f}%")
    if short_count > 0:
        print(f"空头: {short_count}笔, 追踪止盈:{short_wins}笔, 止盈率:{short_wins/short_count*100:.1f}%")


if __name__ == "__main__":
    main()
