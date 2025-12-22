"""
快速参数优化 - 减少组合数量
"""
import sys
import numpy as np
from itertools import product
from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import FOMOConfig

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)


class QuickBacktester:
    def __init__(self, config, use_time_stop=False):
        self.config = config
        self.use_time_stop = use_time_stop

    def run(self, highs, lows, closes, volumes):
        capital = self.config.initial_capital
        campaign = None
        trades = []
        min_bars = max(self.config.breakout_lookback + 10, 70)

        for i in range(min_bars, len(closes)):
            price = closes[i]
            high = highs[i]
            low = lows[i]

            # ATR
            trs = []
            for j in range(max(1, i-14), i+1):
                if j > 0:
                    tr = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
                    trs.append(tr)
            if len(trs) < 14:
                continue
            atr = np.mean(trs[-14:])

            if campaign:
                if high > campaign["highest"]:
                    campaign["highest"] = high

                # 追踪
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

                # 时间止损
                exit_now = False
                if low <= eff_stop:
                    exit_now = True
                elif self.use_time_stop:
                    bars = i - campaign["bar"]
                    if bars >= self.config.time_stop_bars and profit_R < self.config.need_reach_R_in_time:
                        exit_now = True

                if exit_now:
                    exit_p = min(eff_stop, price) * 0.999
                    pnl = (exit_p - campaign["avg"]) * campaign["qty"] - exit_p * campaign["qty"] * 0.0006
                    trades.append(pnl)
                    capital += pnl
                    campaign = None

            if campaign is None and i >= 60:
                highest = max(highs[i-self.config.breakout_lookback:i])
                breakout = highest + self.config.breakout_buffer_atr * atr
                if price <= breakout:
                    continue

                vol_ma = np.mean(volumes[i-60:i])
                if vol_ma <= 0 or volumes[i] / vol_ma < self.config.vol_ratio_entry:
                    continue

                stop_pct = self.config.k_stop_atr * atr / price
                stop_pct = max(self.config.stop_pct_min, min(stop_pct, self.config.stop_pct_max))
                if stop_pct >= self.config.stop_pct_max:
                    continue

                notional = 50.0 / stop_pct
                entry = price * 1.001
                qty = notional / entry
                stop = entry * (1 - stop_pct)
                R = entry - stop
                capital -= notional * 0.0006

                campaign = {"entry": entry, "avg": entry, "stop": stop, "R": R, "qty": qty,
                           "highest": high, "trail_on": False, "trail": None, "bar": i}

        if campaign:
            pnl = (closes[-1] * 0.999 - campaign["avg"]) * campaign["qty"]
            trades.append(pnl)
            capital += pnl

        return sum(trades), len(trades), sum(1 for t in trades if t > 0)


def main():
    print("=" * 50)
    print("快速参数优化")
    print("=" * 50)

    fetcher = MultiPairFetcher()
    print("\n1. 获取数据...")
    hot = fetcher.get_hot_pairs(top_n=5)
    pairs = [p['symbol'] for p in hot]
    print(f"   交易对: {pairs}")

    data = fetcher.fetch_all_pairs_data(pairs, "1m", 3)
    print(f"   完成: {len(data)} 对\n")

    # 第一步：对比时间止损
    print("2. 对比时间止损开/关...")
    cfg = FOMOConfig(vol_ratio_entry=2.0, breakout_lookback=30, breakout_buffer_atr=0.3,
                     k_stop_atr=2.0, trail_k_atr=3.0, enable_trail_after_R=1.5,
                     time_stop_bars=60, need_reach_R_in_time=0.3)

    pnl_on, tr_on, win_on = 0, 0, 0
    pnl_off, tr_off, win_off = 0, 0, 0

    for sym, d in data.items():
        ts, o, h, l, c, v = d
        if len(ts) < 100:
            continue

        bt_on = QuickBacktester(cfg, use_time_stop=True)
        p, t, w = bt_on.run(h, l, c, v)
        pnl_on += p; tr_on += t; win_on += w

        bt_off = QuickBacktester(cfg, use_time_stop=False)
        p, t, w = bt_off.run(h, l, c, v)
        pnl_off += p; tr_off += t; win_off += w

    print(f"   开启: PnL={pnl_on:+.0f}, 交易={tr_on}, 胜率={win_on/tr_on*100:.0f}%")
    print(f"   关闭: PnL={pnl_off:+.0f}, 交易={tr_off}, 胜率={win_off/tr_off*100:.0f}%")
    use_ts = pnl_on > pnl_off
    print(f"   -> {'开启' if use_ts else '关闭'}更好\n")

    # 第二步：优化入场参数
    print("3. 优化入场参数...")
    vol_list = [1.5, 2.0, 3.0, 4.0]
    brk_list = [20, 40, 60]
    buf_list = [0.2, 0.4]
    stop_list = [1.5, 2.0, 2.5, 3.0]

    results = []
    total = len(vol_list) * len(brk_list) * len(buf_list) * len(stop_list)
    cnt = 0

    for vol, brk, buf, stp in product(vol_list, brk_list, buf_list, stop_list):
        cnt += 1
        cfg = FOMOConfig(vol_ratio_entry=vol, breakout_lookback=brk, breakout_buffer_atr=buf,
                         k_stop_atr=stp, trail_k_atr=3.0, enable_trail_after_R=1.5,
                         time_stop_bars=60, need_reach_R_in_time=0.3)

        total_pnl, total_tr, total_win = 0, 0, 0
        for sym, d in data.items():
            ts, o, h, l, c, v = d
            if len(ts) < 100:
                continue
            bt = QuickBacktester(cfg, use_time_stop=use_ts)
            p, t, w = bt.run(h, l, c, v)
            total_pnl += p; total_tr += t; total_win += w

        wr = total_win / total_tr if total_tr > 0 else 0
        results.append((total_pnl, total_tr, wr, vol, brk, buf, stp))

        if cnt % 20 == 0:
            print(f"   进度: {cnt}/{total}")

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n   入场参数 TOP 10:")
    print(f"   {'#':<3} {'Vol':>4} {'Brk':>4} {'Buf':>4} {'Stp':>4} | {'PnL':>8} {'Tr':>4} {'WR':>5}")
    for i, r in enumerate(results[:10], 1):
        print(f"   {i:<3} {r[3]:>4.1f} {r[4]:>4} {r[5]:>4.1f} {r[6]:>4.1f} | {r[0]:>+8.0f} {r[1]:>4} {r[2]*100:>4.0f}%")

    best1 = results[0]

    # 第三步：优化追踪止损
    print("\n4. 优化追踪止损...")
    trail_atr_list = [2.0, 2.5, 3.0, 4.0, 5.0]
    trail_r_list = [0.5, 1.0, 1.5, 2.0]

    results2 = []
    for ta, tr in product(trail_atr_list, trail_r_list):
        cfg = FOMOConfig(vol_ratio_entry=best1[3], breakout_lookback=best1[4],
                         breakout_buffer_atr=best1[5], k_stop_atr=best1[6],
                         trail_k_atr=ta, enable_trail_after_R=tr,
                         time_stop_bars=60, need_reach_R_in_time=0.3)

        total_pnl, total_tr, total_win = 0, 0, 0
        for sym, d in data.items():
            ts, o, h, l, c, v = d
            if len(ts) < 100:
                continue
            bt = QuickBacktester(cfg, use_time_stop=use_ts)
            p, t, w = bt.run(h, l, c, v)
            total_pnl += p; total_tr += t; total_win += w

        wr = total_win / total_tr if total_tr > 0 else 0
        results2.append((total_pnl, total_tr, wr, ta, tr))

    results2.sort(key=lambda x: x[0], reverse=True)

    print(f"   追踪止损 TOP 5:")
    print(f"   {'#':<3} {'TrlATR':>6} {'TrlR':>5} | {'PnL':>8} {'Tr':>4} {'WR':>5}")
    for i, r in enumerate(results2[:5], 1):
        print(f"   {i:<3} {r[3]:>6.1f} {r[4]:>5.1f} | {r[0]:>+8.0f} {r[1]:>4} {r[2]*100:>4.0f}%")

    best2 = results2[0]

    # 最终配置
    print("\n" + "=" * 50)
    print("最佳配置")
    print("=" * 50)
    print(f"""
FOMOConfig(
    # 入场
    vol_ratio_entry = {best1[3]},
    breakout_lookback = {best1[4]},
    breakout_buffer_atr = {best1[5]},

    # 止损
    k_stop_atr = {best1[6]},
    stop_pct_min = 0.01,
    stop_pct_max = 0.06,

    # 追踪止损
    trail_k_atr = {best2[3]},
    enable_trail_after_R = {best2[4]},

    # 时间止损: {'开启' if use_ts else '关闭'}
    time_stop_bars = 60,
    need_reach_R_in_time = 0.3,
)

预期表现:
  PnL: {best2[0]:+.2f}
  交易数: {best2[1]}
  胜率: {best2[2]*100:.1f}%
""")


if __name__ == "__main__":
    main()
