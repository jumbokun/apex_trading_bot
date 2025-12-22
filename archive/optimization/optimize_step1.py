"""
第1步：获取数据 + 测试开启/关闭时间止损
"""
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import FOMOConfig, BacktestResult, Campaign, Trade, TechnicalIndicators


class FlexibleBacktester:
    """灵活的回测器，支持开关时间止损"""

    def __init__(self, config: FOMOConfig, use_time_stop: bool = True):
        self.config = config
        self.use_time_stop = use_time_stop
        self.indicators = TechnicalIndicators()

    def run_backtest(self, timestamps, opens, highs, lows, closes, volumes, symbol="BTC-USDC"):
        capital = self.config.initial_capital
        campaign = None
        trades = []

        min_bars = max(self.config.atr_len + 10, self.config.breakout_lookback + 10, 70)

        for i in range(min_bars, len(closes)):
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            atr = self._calc_atr(highs[:i+1], lows[:i+1], closes[:i+1])
            if not atr:
                continue

            if campaign:
                should_exit = False
                exit_reason = ""
                exit_price = current_price

                if current_high > campaign["highest"]:
                    campaign["highest"] = current_high

                # 追踪止损
                if campaign["trail_enabled"]:
                    new_trail = campaign["highest"] - self.config.trail_k_atr * atr
                    if campaign["trail_stop"] is None or new_trail > campaign["trail_stop"]:
                        campaign["trail_stop"] = new_trail

                # 启用追踪
                profit_R = (current_price - campaign["entry"]) / campaign["R"]
                if not campaign["trail_enabled"] and profit_R >= self.config.enable_trail_after_R:
                    campaign["trail_enabled"] = True
                    campaign["trail_stop"] = campaign["highest"] - self.config.trail_k_atr * atr

                effective_stop = campaign["stop"]
                if campaign["trail_stop"] and campaign["trail_stop"] > effective_stop:
                    effective_stop = campaign["trail_stop"]

                # 止损检查
                if current_low <= effective_stop:
                    should_exit = True
                    exit_price = effective_stop
                    exit_reason = "Trailing" if campaign["trail_stop"] and effective_stop == campaign["trail_stop"] else "Stop"

                # 时间止损
                if not should_exit and self.use_time_stop:
                    bars_held = i - campaign["bar_idx"]
                    if bars_held >= self.config.time_stop_bars:
                        if profit_R < self.config.need_reach_R_in_time:
                            should_exit = True
                            exit_reason = "TimeStop"

                if should_exit:
                    exit_price *= (1 - self.config.slippage)
                    pnl = (exit_price - campaign["avg"]) * campaign["qty"]
                    pnl -= exit_price * campaign["qty"] * self.config.commission_rate
                    r_mult = (exit_price - campaign["entry"]) / campaign["R"]

                    trades.append({
                        "pnl": pnl,
                        "r": r_mult,
                        "reason": exit_reason
                    })
                    capital += pnl
                    campaign = None

            # 新入场
            if campaign is None:
                # 突破检查
                if i < self.config.breakout_lookback:
                    continue
                highest = max(highs[i-self.config.breakout_lookback:i])
                breakout_level = highest + self.config.breakout_buffer_atr * atr
                if current_price <= breakout_level:
                    continue

                # 成交量检查
                if i < 60:
                    continue
                vol_ma = np.mean(volumes[i-60:i])
                vol_ratio = volumes[i] / vol_ma if vol_ma > 0 else 0
                if vol_ratio < self.config.vol_ratio_entry:
                    continue

                # 计算止损
                stop_pct = self.config.k_stop_atr * atr / current_price
                if stop_pct < self.config.stop_pct_min:
                    stop_pct = self.config.stop_pct_min
                elif stop_pct > self.config.stop_pct_max:
                    continue

                notional = self.config.risk_cash_per_campaign / stop_pct
                entry_price = current_price * (1 + self.config.slippage)
                qty = notional / entry_price
                stop_price = entry_price * (1 - stop_pct)
                R_price = entry_price - stop_price

                capital -= notional * self.config.commission_rate

                campaign = {
                    "entry": entry_price,
                    "avg": entry_price,
                    "stop": stop_price,
                    "R": R_price,
                    "qty": qty,
                    "highest": current_high,
                    "bar_idx": i,
                    "trail_enabled": False,
                    "trail_stop": None
                }

        # 关闭剩余仓位
        if campaign:
            final_price = closes[-1] * (1 - self.config.slippage)
            pnl = (final_price - campaign["avg"]) * campaign["qty"]
            r_mult = (final_price - campaign["entry"]) / campaign["R"]
            trades.append({"pnl": pnl, "r": r_mult, "reason": "End"})
            capital += pnl

        return trades, capital

    def _calc_atr(self, highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return None
        trs = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return np.mean(trs[-period:]) if len(trs) >= period else None


def run_test(config, all_data, use_time_stop):
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    exit_reasons = {}

    for symbol, data in all_data.items():
        timestamps, opens, highs, lows, closes, volumes = data
        if len(timestamps) < 100:
            continue

        bt = FlexibleBacktester(config, use_time_stop)
        trades, capital = bt.run_backtest(timestamps, opens, highs, lows, closes, volumes, symbol)

        for t in trades:
            total_trades += 1
            total_pnl += t["pnl"]
            if t["pnl"] > 0:
                total_wins += 1
            r = t["reason"]
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

    win_rate = total_wins / total_trades if total_trades > 0 else 0
    return {
        "pnl": total_pnl,
        "trades": total_trades,
        "win_rate": win_rate,
        "exits": exit_reasons
    }


def main():
    print("=" * 60)
    print("第1步：获取数据 + 对比时间止损开/关")
    print("=" * 60)

    # 获取数据
    fetcher = MultiPairFetcher()
    print("\n获取热门交易对...")
    hot_pairs = fetcher.get_hot_pairs(top_n=6)
    pairs = [p.get('symbol') for p in hot_pairs]
    print(f"交易对: {pairs}")

    print("\n获取3天数据...")
    all_data = fetcher.fetch_all_pairs_data(pairs, "1m", 3)
    print(f"获取到 {len(all_data)} 个交易对数据")

    # 保存数据供后续使用
    # (数据太大，不保存，每步重新获取)

    # 基础配置
    config = FOMOConfig(
        initial_capital=5000.0,
        risk_cash_per_campaign=50.0,
        atr_len=14,
        vol_ratio_entry=2.0,
        vol_ratio_add=1.5,
        breakout_lookback=30,
        breakout_buffer_atr=0.3,
        k_stop_atr=2.0,
        stop_pct_min=0.01,
        stop_pct_max=0.06,
        trail_k_atr=3.0,
        enable_trail_after_R=1.5,
        time_stop_bars=60,
        need_reach_R_in_time=0.3,
    )

    # 测试开启时间止损
    print("\n测试开启时间止损...")
    result_on = run_test(config, all_data, use_time_stop=True)
    print(f"  PnL: {result_on['pnl']:+.2f} | 交易: {result_on['trades']} | 胜率: {result_on['win_rate']:.1%}")
    print(f"  出场: {result_on['exits']}")

    # 测试关闭时间止损
    print("\n测试关闭时间止损...")
    result_off = run_test(config, all_data, use_time_stop=False)
    print(f"  PnL: {result_off['pnl']:+.2f} | 交易: {result_off['trades']} | 胜率: {result_off['win_rate']:.1%}")
    print(f"  出场: {result_off['exits']}")

    # 结论
    better = "开启" if result_on['pnl'] > result_off['pnl'] else "关闭"
    print(f"\n结论: {better}时间止损更好")
    print(f"  开启: {result_on['pnl']:+.2f}")
    print(f"  关闭: {result_off['pnl']:+.2f}")

    # 保存结论
    with open("d:/1desktop/web3work/apex_trading_bot/opt_result_step1.json", "w") as f:
        json.dump({
            "use_time_stop": result_on['pnl'] > result_off['pnl'],
            "result_on": result_on,
            "result_off": result_off,
        }, f)
    print("\n结果已保存到 opt_result_step1.json")


if __name__ == "__main__":
    main()
