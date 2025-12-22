"""
第2步：优化入场参数 (关闭时间止损)
基于第1步结论：关闭时间止损更好
"""
import numpy as np
from itertools import product
from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import FOMOConfig


class SimpleBacktester:
    def __init__(self, config):
        self.config = config

    def run(self, timestamps, opens, highs, lows, closes, volumes):
        capital = self.config.initial_capital
        campaign = None
        trades = []
        min_bars = max(self.config.atr_len + 10, self.config.breakout_lookback + 10, 70)

        for i in range(min_bars, len(closes)):
            price = closes[i]
            high = highs[i]
            low = lows[i]

            atr = self._atr(highs[:i+1], lows[:i+1], closes[:i+1])
            if not atr:
                continue

            if campaign:
                if high > campaign["highest"]:
                    campaign["highest"] = high

                # 追踪止损
                if campaign["trail_on"]:
                    new_trail = campaign["highest"] - self.config.trail_k_atr * atr
                    if campaign["trail"] is None or new_trail > campaign["trail"]:
                        campaign["trail"] = new_trail

                profit_R = (price - campaign["entry"]) / campaign["R"]
                if not campaign["trail_on"] and profit_R >= self.config.enable_trail_after_R:
                    campaign["trail_on"] = True
                    campaign["trail"] = campaign["highest"] - self.config.trail_k_atr * atr

                eff_stop = campaign["stop"]
                if campaign["trail"] and campaign["trail"] > eff_stop:
                    eff_stop = campaign["trail"]

                if low <= eff_stop:
                    exit_p = eff_stop * (1 - self.config.slippage)
                    pnl = (exit_p - campaign["avg"]) * campaign["qty"]
                    pnl -= exit_p * campaign["qty"] * self.config.commission_rate
                    trades.append(pnl)
                    capital += pnl
                    campaign = None

            if campaign is None:
                if i < self.config.breakout_lookback or i < 60:
                    continue

                highest = max(highs[i-self.config.breakout_lookback:i])
                breakout = highest + self.config.breakout_buffer_atr * atr
                if price <= breakout:
                    continue

                vol_ma = np.mean(volumes[i-60:i])
                vol_ratio = volumes[i] / vol_ma if vol_ma > 0 else 0
                if vol_ratio < self.config.vol_ratio_entry:
                    continue

                stop_pct = self.config.k_stop_atr * atr / price
                if stop_pct < self.config.stop_pct_min:
                    stop_pct = self.config.stop_pct_min
                elif stop_pct > self.config.stop_pct_max:
                    continue

                notional = self.config.risk_cash_per_campaign / stop_pct
                entry = price * (1 + self.config.slippage)
                qty = notional / entry
                stop = entry * (1 - stop_pct)
                R = entry - stop

                capital -= notional * self.config.commission_rate

                campaign = {
                    "entry": entry, "avg": entry, "stop": stop, "R": R, "qty": qty,
                    "highest": high, "trail_on": False, "trail": None
                }

        if campaign:
            final = closes[-1] * (1 - self.config.slippage)
            pnl = (final - campaign["avg"]) * campaign["qty"]
            trades.append(pnl)
            capital += pnl

        return trades, capital

    def _atr(self, highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return None
        trs = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return np.mean(trs[-period:]) if len(trs) >= period else None


def test_config(config, all_data):
    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for symbol, data in all_data.items():
        ts, o, h, l, c, v = data
        if len(ts) < 100:
            continue
        bt = SimpleBacktester(config)
        trades, cap = bt.run(ts, o, h, l, c, v)
        for pnl in trades:
            total_trades += 1
            total_pnl += pnl
            if pnl > 0:
                total_wins += 1

    return total_pnl, total_trades, total_wins / total_trades if total_trades > 0 else 0


def main():
    print("=" * 60)
    print("第2步：优化入场参数")
    print("=" * 60)

    fetcher = MultiPairFetcher()
    print("\n获取数据...")
    hot_pairs = fetcher.get_hot_pairs(top_n=6)
    pairs = [p.get('symbol') for p in hot_pairs]
    all_data = fetcher.fetch_all_pairs_data(pairs, "1m", 3)
    print(f"数据就绪: {len(all_data)} 对")

    # 参数网格
    vol_list = [1.5, 2.0, 2.5, 3.0, 4.0]
    brk_list = [20, 30, 40, 60]
    buf_list = [0.2, 0.3, 0.4, 0.5]
    stop_list = [1.5, 2.0, 2.5, 3.0]

    total = len(vol_list) * len(brk_list) * len(buf_list) * len(stop_list)
    print(f"\n测试 {total} 个参数组合...")

    results = []
    count = 0

    for vol, brk, buf, stop in product(vol_list, brk_list, buf_list, stop_list):
        count += 1
        config = FOMOConfig(
            initial_capital=5000.0,
            risk_cash_per_campaign=50.0,
            atr_len=14,
            vol_ratio_entry=vol,
            breakout_lookback=brk,
            breakout_buffer_atr=buf,
            k_stop_atr=stop,
            stop_pct_min=0.01,
            stop_pct_max=0.06,
            trail_k_atr=3.0,
            enable_trail_after_R=1.5,
        )
        pnl, trades, wr = test_config(config, all_data)
        results.append((pnl, trades, wr, vol, brk, buf, stop))

        if count % 50 == 0:
            print(f"  进度: {count}/{total}")

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 60)
    print("入场参数 TOP 15")
    print("=" * 60)
    print(f"{'排名':<4} {'Vol':>5} {'Brk':>5} {'Buf':>5} {'Stop':>5} | {'PnL':>10} {'Trades':>7} {'WinR':>7}")
    print("-" * 60)

    for i, (pnl, trades, wr, vol, brk, buf, stop) in enumerate(results[:15], 1):
        print(f"{i:<4} {vol:>5.1f} {brk:>5} {buf:>5.1f} {stop:>5.1f} | {pnl:>+10.2f} {trades:>7} {wr:>6.1%}")

    best = results[0]
    print(f"\n最佳入场参数:")
    print(f"  vol_ratio_entry = {best[3]}")
    print(f"  breakout_lookback = {best[4]}")
    print(f"  breakout_buffer_atr = {best[5]}")
    print(f"  k_stop_atr = {best[6]}")
    print(f"  PnL = {best[0]:+.2f}, Trades = {best[1]}, WinRate = {best[2]:.1%}")


if __name__ == "__main__":
    main()
