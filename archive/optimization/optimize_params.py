"""
FOMO策略参数优化
通过网格搜索找出当前市场条件下最优参数组合
包括测试开启/关闭时间止损
"""
import sys
import io
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from itertools import product
from dataclasses import dataclass, field
import numpy as np

from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import FOMOConfig, BacktestResult, Campaign, Trade, TechnicalIndicators


class FlexibleBacktester:
    """灵活的回测器，支持开关时间止损"""

    def __init__(self, config: FOMOConfig, use_time_stop: bool = True):
        self.config = config
        self.use_time_stop = use_time_stop
        self.indicators = TechnicalIndicators()

    def calculate_stop_pct(self, atr: float, price: float) -> Tuple[float, bool]:
        stop_dist = self.config.k_stop_atr * atr
        stop_pct = stop_dist / price

        if stop_pct < self.config.stop_pct_min:
            stop_pct = self.config.stop_pct_min
        elif stop_pct > self.config.stop_pct_max:
            return stop_pct, False

        return stop_pct, True

    def calculate_leverage(self, stop_pct: float) -> int:
        L_cap = int(1 / (4 * stop_pct))
        L = max(self.config.leverage_min, min(L_cap, self.config.leverage_max))
        return L

    def calculate_initial_position(self, price: float, stop_pct: float) -> Tuple[float, float]:
        notional = self.config.risk_cash_per_campaign / stop_pct
        qty = notional / price
        return notional, qty

    def calculate_risk_stop(self, avg_entry: float, total_qty: float) -> float:
        if total_qty <= 0:
            return 0
        return avg_entry - (self.config.risk_cash_per_campaign / total_qty)

    def check_volume_confirmation(self, volumes: List[float], current_idx: int,
                                   ma_period: int = 60, required_ratio: float = 2.0) -> Tuple[bool, float]:
        if current_idx < ma_period:
            return False, 0

        vol_ma = np.mean(volumes[current_idx - ma_period:current_idx])
        current_vol = volumes[current_idx]
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 0

        return vol_ratio >= required_ratio, vol_ratio

    def check_breakout(self, highs: List[float], closes: List[float],
                       current_idx: int, atr: float) -> Tuple[bool, float]:
        lookback = self.config.breakout_lookback
        if current_idx < lookback:
            return False, 0

        recent_highs = highs[current_idx - lookback:current_idx]
        highest_high = max(recent_highs)

        buffer = self.config.breakout_buffer_atr * atr
        breakout_level = highest_high + buffer

        current_close = closes[current_idx]
        is_breakout = current_close > breakout_level

        return is_breakout, breakout_level

    def should_add_position(self, campaign: Campaign, current_price: float,
                           atr: float, vol_ratio: float, ema9: float) -> bool:
        if campaign.add_count >= self.config.max_adds:
            return False

        next_add_idx = campaign.add_count
        if next_add_idx >= len(self.config.add_levels_R):
            return False

        required_R = self.config.add_levels_R[next_add_idx]
        target_price = campaign.initial_entry_price + required_R * campaign.R_price

        if current_price < target_price:
            return False

        if vol_ratio < self.config.vol_ratio_add:
            return False

        if ema9 and current_price < ema9:
            return False

        add_frac = self.config.add_notional_fracs[next_add_idx]
        add_notional = campaign.total_notional * add_frac / (1 + sum(self.config.add_notional_fracs[:next_add_idx]))
        add_qty = add_notional / current_price

        new_total_qty = campaign.total_qty + add_qty
        new_avg = (campaign.avg_entry * campaign.total_qty + current_price * add_qty) / new_total_qty
        new_risk_stop = self.calculate_risk_stop(new_avg, new_total_qty)

        stop_distance = current_price - max(new_risk_stop, campaign.current_stop)
        if stop_distance < self.config.require_stop_distance_atr * atr:
            return False

        return True

    def run_backtest(
        self,
        timestamps: List[datetime],
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        symbol: str = "BTC-USDC"
    ) -> BacktestResult:

        capital = self.config.initial_capital
        campaign: Optional[Campaign] = None
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []

        daily_pnl = 0.0
        consecutive_losses = 0
        cooldown_until = 0
        symbol_cooldown: Dict[str, int] = {}

        peak_equity = capital
        max_drawdown = 0
        max_drawdown_pct = 0
        max_consec_losses_seen = 0

        min_bars = max(self.config.atr_len + 10, self.config.breakout_lookback + 10, 70)

        for i in range(min_bars, len(closes)):
            current_time = timestamps[i]
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            if i > 0 and timestamps[i].date() != timestamps[i-1].date():
                daily_pnl = 0.0

            atr = self.indicators.calculate_atr(
                highs[:i+1], lows[:i+1], closes[:i+1], self.config.atr_len
            )
            if not atr:
                continue

            ema9 = self.indicators.calculate_ema(closes[:i+1], 9)

            if campaign:
                unrealized_pnl = (current_price - campaign.avg_entry) * campaign.total_qty
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity_curve.append((current_time, current_equity))

            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = peak_equity - current_equity
            drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

            if campaign:
                should_exit = False
                exit_reason = ""
                exit_price = current_price

                if current_high > campaign.highest_since_entry:
                    campaign.highest_since_entry = current_high

                if campaign.trail_enabled:
                    new_trail = campaign.highest_since_entry - self.config.trail_k_atr * atr
                    if campaign.trail_stop is None or new_trail > campaign.trail_stop:
                        campaign.trail_stop = new_trail

                if not campaign.trail_enabled:
                    profit_R = (current_price - campaign.initial_entry_price) / campaign.R_price
                    if profit_R >= self.config.enable_trail_after_R:
                        campaign.trail_enabled = True
                        campaign.trail_stop = campaign.highest_since_entry - self.config.trail_k_atr * atr

                effective_stop = campaign.current_stop
                if campaign.trail_stop and campaign.trail_stop > effective_stop:
                    effective_stop = campaign.trail_stop

                if current_low <= effective_stop:
                    should_exit = True
                    exit_price = effective_stop
                    if campaign.trail_stop and effective_stop == campaign.trail_stop:
                        exit_reason = f"Trailing Stop"
                    else:
                        exit_reason = f"Stop Loss"

                # 时间止损 - 可以开关
                if not should_exit and self.use_time_stop:
                    bars_held = i - campaign.entry_bar_idx
                    if bars_held >= self.config.time_stop_bars:
                        profit_R = (current_price - campaign.initial_entry_price) / campaign.R_price
                        if profit_R < self.config.need_reach_R_in_time:
                            should_exit = True
                            exit_reason = f"Time Stop"

                if not should_exit:
                    vol_confirmed, vol_ratio = self.check_volume_confirmation(
                        volumes, i, 60, self.config.vol_ratio_add
                    )

                    if self.should_add_position(campaign, current_price, atr, vol_ratio, ema9):
                        add_idx = campaign.add_count
                        add_frac = self.config.add_notional_fracs[add_idx]
                        initial_notional = campaign.entries[0][1] * campaign.entries[0][2]
                        add_notional = initial_notional * add_frac
                        potential_total = campaign.total_notional + add_notional
                        max_allowed = initial_notional * self.config.max_total_notional_mult

                        if potential_total <= max_allowed:
                            add_price = current_price * (1 + self.config.slippage)
                            add_qty = add_notional / add_price
                            commission = add_notional * self.config.commission_rate
                            capital -= commission

                            new_total_qty = campaign.total_qty + add_qty
                            new_avg = (campaign.avg_entry * campaign.total_qty + add_price * add_qty) / new_total_qty

                            campaign.entries.append((current_time, add_price, add_qty))
                            campaign.total_qty = new_total_qty
                            campaign.avg_entry = new_avg
                            campaign.total_notional += add_notional
                            campaign.add_count += 1

                            new_risk_stop = self.calculate_risk_stop(new_avg, new_total_qty)
                            campaign.current_stop = max(campaign.current_stop, new_risk_stop)

                if should_exit:
                    exit_price = exit_price * (1 - self.config.slippage)
                    pnl = (exit_price - campaign.avg_entry) * campaign.total_qty
                    commission = exit_price * campaign.total_qty * self.config.commission_rate
                    pnl -= commission

                    pnl_pct = pnl / (campaign.avg_entry * campaign.total_qty)
                    r_multiple = (exit_price - campaign.initial_entry_price) / campaign.R_price

                    trade = Trade(
                        symbol=symbol,
                        side=campaign.side,
                        entry_time=campaign.entries[0][0],
                        exit_time=current_time,
                        avg_entry_price=campaign.avg_entry,
                        exit_price=exit_price,
                        total_qty=campaign.total_qty,
                        add_count=campaign.add_count,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        r_multiple=r_multiple
                    )
                    trades.append(trade)

                    capital += pnl
                    daily_pnl += pnl

                    if pnl < 0:
                        consecutive_losses += 1
                        if consecutive_losses > max_consec_losses_seen:
                            max_consec_losses_seen = consecutive_losses
                        if consecutive_losses >= self.config.max_consecutive_losses:
                            cooldown_until = i + self.config.cooldown_bars
                        symbol_cooldown[symbol] = i + self.config.same_symbol_cooldown_bars
                    else:
                        consecutive_losses = 0

                    campaign = None

            if campaign is None:
                if daily_pnl <= -self.config.daily_loss_limit:
                    continue
                if i < cooldown_until:
                    continue
                if symbol in symbol_cooldown and i < symbol_cooldown[symbol]:
                    continue

                is_breakout, breakout_level = self.check_breakout(highs, closes, i, atr)
                if not is_breakout:
                    continue

                vol_confirmed, vol_ratio = self.check_volume_confirmation(
                    volumes, i, 60, self.config.vol_ratio_entry
                )
                if not vol_confirmed:
                    continue

                stop_pct, is_valid = self.calculate_stop_pct(atr, current_price)
                if not is_valid:
                    continue

                notional, qty = self.calculate_initial_position(current_price, stop_pct)

                leverage = self.calculate_leverage(stop_pct)
                required_margin = notional / leverage
                if required_margin > capital * 0.5:
                    continue

                entry_price = current_price * (1 + self.config.slippage)
                qty = notional / entry_price
                stop_price = entry_price * (1 - stop_pct)
                R_price = entry_price - stop_price

                commission = notional * self.config.commission_rate
                capital -= commission

                campaign = Campaign(
                    symbol=symbol,
                    side="LONG",
                    initial_entry_price=entry_price,
                    initial_stop=stop_price,
                    R_price=R_price,
                    current_stop=stop_price,
                    highest_since_entry=current_high,
                    entry_bar_idx=i,
                    entry_atr=atr,
                    total_qty=qty,
                    avg_entry=entry_price,
                    total_notional=notional
                )
                campaign.entries.append((current_time, entry_price, qty))

        if campaign:
            final_price = closes[-1] * (1 - self.config.slippage)
            pnl = (final_price - campaign.avg_entry) * campaign.total_qty
            commission = final_price * campaign.total_qty * self.config.commission_rate
            pnl -= commission

            r_multiple = (final_price - campaign.initial_entry_price) / campaign.R_price

            trade = Trade(
                symbol=symbol,
                side=campaign.side,
                entry_time=campaign.entries[0][0],
                exit_time=timestamps[-1],
                avg_entry_price=campaign.avg_entry,
                exit_price=final_price,
                total_qty=campaign.total_qty,
                add_count=campaign.add_count,
                pnl=pnl,
                pnl_pct=pnl / (campaign.avg_entry * campaign.total_qty),
                exit_reason="End of Backtest",
                r_multiple=r_multiple
            )
            trades.append(trade)
            capital += pnl

        total_return = capital - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        avg_win_R = np.mean([t.r_multiple for t in winning_trades]) if winning_trades else 0
        avg_loss_R = np.mean([t.r_multiple for t in losing_trades]) if losing_trades else 0

        win_rate = len(winning_trades) / len(trades) if trades else 0
        expectancy_R = (win_rate * avg_win_R) + ((1 - win_rate) * avg_loss_R)

        avg_adds = np.mean([t.add_count for t in trades]) if trades else 0

        result = BacktestResult(
            initial_capital=self.config.initial_capital,
            final_capital=capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            profit_factor=profit_factor,
            avg_win_R=avg_win_R,
            avg_loss_R=avg_loss_R,
            expectancy_R=expectancy_R,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            max_consecutive_losses=max_consec_losses_seen,
            avg_adds_per_trade=avg_adds,
            trades=trades,
            equity_curve=equity_curve
        )

        return result


def run_single_backtest(config: FOMOConfig, all_data: Dict, use_time_stop: bool = True) -> Dict:
    """运行单次回测，返回汇总结果"""
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    all_trades = []

    for symbol, data in all_data.items():
        timestamps, opens, highs, lows, closes, volumes = data

        if len(timestamps) < 100:
            continue

        backtester = FlexibleBacktester(config, use_time_stop=use_time_stop)

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

            total_pnl += result.total_return
            total_trades += result.total_trades
            total_wins += result.winning_trades
            all_trades.extend(result.trades)

        except Exception as e:
            continue

    win_rate = total_wins / total_trades if total_trades > 0 else 0
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0

    # 计算期望R
    if all_trades:
        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        avg_win_r = np.mean([t.r_multiple for t in wins]) if wins else 0
        avg_loss_r = np.mean([t.r_multiple for t in losses]) if losses else 0
        expectancy = (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r)

        # 计算出场原因分布
        exit_reasons = {}
        for t in all_trades:
            reason = t.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1
    else:
        expectancy = 0
        avg_win_r = 0
        avg_loss_r = 0
        exit_reasons = {}

    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "exit_reasons": exit_reasons,
    }


def optimize_parameters():
    """网格搜索优化参数"""

    print("=" * 70)
    print("FOMO策略参数优化")
    print("包括测试开启/关闭时间止损")
    print("=" * 70)

    # 获取数据
    fetcher = MultiPairFetcher()
    print("\n[1] 获取热门交易对...")
    hot_pairs = fetcher.get_hot_pairs(top_n=8)
    pairs = [p.get('symbol') for p in hot_pairs]

    print(f"\n[2] 获取3天1分钟数据...")
    all_data = fetcher.fetch_all_pairs_data(pairs, "1m", 3)

    if not all_data:
        print("错误：无法获取数据")
        return

    print(f"\n[3] 开始参数优化...")

    # ========================================
    # 第0轮：对比开启/关闭时间止损
    # ========================================
    print("\n" + "=" * 70)
    print("第0轮：对比开启/关闭时间止损")
    print("=" * 70)

    base_config = FOMOConfig(
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
        leverage_min=2,
        leverage_max=5,
        max_adds=3,
        add_levels_R=[1.5, 2.5, 3.5],
        add_notional_fracs=[0.6, 0.4, 0.3],
        require_stop_distance_atr=1.0,
        trail_k_atr=3.0,
        enable_trail_after_R=1.5,
        time_stop_bars=60,
        need_reach_R_in_time=0.3,
        daily_loss_limit=150.0,
        max_consecutive_losses=5,
        cooldown_bars=30,
        same_symbol_cooldown_bars=60,
        commission_rate=0.0006,
        slippage=0.001,
    )

    # 开启时间止损
    result_with_time = run_single_backtest(base_config, all_data, use_time_stop=True)
    # 关闭时间止损
    result_no_time = run_single_backtest(base_config, all_data, use_time_stop=False)

    print(f"\n开启时间止损:")
    print(f"  PnL: {result_with_time['total_pnl']:+.2f} | Trades: {result_with_time['total_trades']} | "
          f"WinR: {result_with_time['win_rate']:.1%} | Exp: {result_with_time['expectancy']:+.2f}R")
    print(f"  出场原因: {result_with_time['exit_reasons']}")

    print(f"\n关闭时间止损:")
    print(f"  PnL: {result_no_time['total_pnl']:+.2f} | Trades: {result_no_time['total_trades']} | "
          f"WinR: {result_no_time['win_rate']:.1%} | Exp: {result_no_time['expectancy']:+.2f}R")
    print(f"  出场原因: {result_no_time['exit_reasons']}")

    # 决定是否使用时间止损
    use_time_stop_optimal = result_with_time['total_pnl'] > result_no_time['total_pnl']
    print(f"\n结论: {'开启' if use_time_stop_optimal else '关闭'}时间止损更好")

    # ========================================
    # 第1轮：优化入场和止损参数
    # ========================================
    print("\n" + "=" * 70)
    print("第1轮：优化入场和止损参数")
    print("=" * 70)

    results_round1 = []

    vol_configs = [1.5, 2.0, 2.5, 3.0, 4.0]
    breakout_configs = [20, 30, 40, 60]
    buffer_configs = [0.2, 0.3, 0.4, 0.5]
    stop_atr_configs = [1.5, 2.0, 2.5, 3.0]

    test_count = 0
    total_tests = len(vol_configs) * len(breakout_configs) * len(buffer_configs) * len(stop_atr_configs)

    for vol_ratio, breakout, buffer, stop_atr in product(vol_configs, breakout_configs, buffer_configs, stop_atr_configs):
        test_count += 1

        config = FOMOConfig(
            initial_capital=5000.0,
            risk_cash_per_campaign=50.0,
            atr_len=14,

            vol_ratio_entry=vol_ratio,
            vol_ratio_add=1.5,
            breakout_lookback=breakout,
            breakout_buffer_atr=buffer,

            k_stop_atr=stop_atr,
            stop_pct_min=0.01,
            stop_pct_max=0.06,

            leverage_min=2,
            leverage_max=5,

            max_adds=3,
            add_levels_R=[1.5, 2.5, 3.5],
            add_notional_fracs=[0.6, 0.4, 0.3],
            require_stop_distance_atr=1.0,

            trail_k_atr=3.0,
            enable_trail_after_R=1.5,

            time_stop_bars=60,
            need_reach_R_in_time=0.3,

            daily_loss_limit=150.0,
            max_consecutive_losses=5,
            cooldown_bars=30,
            same_symbol_cooldown_bars=60,

            commission_rate=0.0006,
            slippage=0.001,
        )

        result = run_single_backtest(config, all_data, use_time_stop=use_time_stop_optimal)

        results_round1.append({
            "vol_ratio": vol_ratio,
            "breakout": breakout,
            "buffer": buffer,
            "stop_atr": stop_atr,
            **result
        })

        if test_count % 50 == 0:
            print(f"  进度: {test_count}/{total_tests} ({test_count/total_tests*100:.1f}%)")

    results_round1.sort(key=lambda x: x["total_pnl"], reverse=True)

    print(f"\n第1轮前10名:")
    print("-" * 100)
    print(f"{'排名':<4} {'Vol':>5} {'Brk':>5} {'Buf':>5} {'StpA':>6} | {'PnL':>10} {'Trades':>7} {'WinR':>7} {'Exp(R)':>8}")
    print("-" * 100)

    for i, r in enumerate(results_round1[:10], 1):
        print(f"{i:<4} {r['vol_ratio']:>5.1f} {r['breakout']:>5} {r['buffer']:>5.1f} {r['stop_atr']:>6.1f} | "
              f"{r['total_pnl']:>+10.2f} {r['total_trades']:>7} {r['win_rate']:>6.1%} {r['expectancy']:>+8.2f}R")

    best_r1 = results_round1[0]

    # ========================================
    # 第2轮：优化追踪止损参数
    # ========================================
    print("\n" + "=" * 70)
    print("第2轮：优化追踪止损参数")
    print("=" * 70)

    results_round2 = []

    trail_atr_configs = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    trail_after_configs = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]

    for trail_atr, trail_after in product(trail_atr_configs, trail_after_configs):
        config = FOMOConfig(
            initial_capital=5000.0,
            risk_cash_per_campaign=50.0,
            atr_len=14,

            vol_ratio_entry=best_r1["vol_ratio"],
            vol_ratio_add=1.5,
            breakout_lookback=best_r1["breakout"],
            breakout_buffer_atr=best_r1["buffer"],

            k_stop_atr=best_r1["stop_atr"],
            stop_pct_min=0.01,
            stop_pct_max=0.06,

            leverage_min=2,
            leverage_max=5,

            max_adds=3,
            add_levels_R=[1.5, 2.5, 3.5],
            add_notional_fracs=[0.6, 0.4, 0.3],
            require_stop_distance_atr=1.0,

            trail_k_atr=trail_atr,
            enable_trail_after_R=trail_after,

            time_stop_bars=60,
            need_reach_R_in_time=0.3,

            daily_loss_limit=150.0,
            max_consecutive_losses=5,
            cooldown_bars=30,
            same_symbol_cooldown_bars=60,

            commission_rate=0.0006,
            slippage=0.001,
        )

        result = run_single_backtest(config, all_data, use_time_stop=use_time_stop_optimal)

        results_round2.append({
            "trail_atr": trail_atr,
            "trail_after": trail_after,
            **result
        })

    results_round2.sort(key=lambda x: x["total_pnl"], reverse=True)

    print(f"\n第2轮前10名:")
    print("-" * 90)
    print(f"{'排名':<4} {'TrailATR':>9} {'TrailR':>8} | {'PnL':>10} {'Trades':>7} {'WinR':>7} {'Exp(R)':>8}")
    print("-" * 90)

    for i, r in enumerate(results_round2[:10], 1):
        print(f"{i:<4} {r['trail_atr']:>9.1f} {r['trail_after']:>8.1f} | "
              f"{r['total_pnl']:>+10.2f} {r['total_trades']:>7} {r['win_rate']:>6.1%} {r['expectancy']:>+8.2f}R")

    best_r2 = results_round2[0]

    # ========================================
    # 第3轮：如果使用时间止损，优化时间止损参数
    # ========================================
    if use_time_stop_optimal:
        print("\n" + "=" * 70)
        print("第3轮：优化时间止损参数")
        print("=" * 70)

        results_round3 = []

        time_configs = [30, 45, 60, 90, 120, 180, 240]
        need_r_configs = [0.1, 0.2, 0.3, 0.4, 0.5]

        for time_bars, need_r in product(time_configs, need_r_configs):
            config = FOMOConfig(
                initial_capital=5000.0,
                risk_cash_per_campaign=50.0,
                atr_len=14,

                vol_ratio_entry=best_r1["vol_ratio"],
                vol_ratio_add=1.5,
                breakout_lookback=best_r1["breakout"],
                breakout_buffer_atr=best_r1["buffer"],

                k_stop_atr=best_r1["stop_atr"],
                stop_pct_min=0.01,
                stop_pct_max=0.06,

                leverage_min=2,
                leverage_max=5,

                max_adds=3,
                add_levels_R=[1.5, 2.5, 3.5],
                add_notional_fracs=[0.6, 0.4, 0.3],
                require_stop_distance_atr=1.0,

                trail_k_atr=best_r2["trail_atr"],
                enable_trail_after_R=best_r2["trail_after"],

                time_stop_bars=time_bars,
                need_reach_R_in_time=need_r,

                daily_loss_limit=150.0,
                max_consecutive_losses=5,
                cooldown_bars=30,
                same_symbol_cooldown_bars=60,

                commission_rate=0.0006,
                slippage=0.001,
            )

            result = run_single_backtest(config, all_data, use_time_stop=True)

            results_round3.append({
                "time_bars": time_bars,
                "need_r": need_r,
                **result
            })

        results_round3.sort(key=lambda x: x["total_pnl"], reverse=True)

        print(f"\n第3轮前10名:")
        print("-" * 90)
        print(f"{'排名':<4} {'TimeBars':>9} {'NeedR':>7} | {'PnL':>10} {'Trades':>7} {'WinR':>7} {'Exp(R)':>8}")
        print("-" * 90)

        for i, r in enumerate(results_round3[:10], 1):
            print(f"{i:<4} {r['time_bars']:>9} {r['need_r']:>7.1f} | "
                  f"{r['total_pnl']:>+10.2f} {r['total_trades']:>7} {r['win_rate']:>6.1%} {r['expectancy']:>+8.2f}R")

        best_r3 = results_round3[0]
        best_time_bars = best_r3["time_bars"]
        best_need_r = best_r3["need_r"]
    else:
        best_time_bars = 60
        best_need_r = 0.3

    # ========================================
    # 第4轮：最终微调
    # ========================================
    print("\n" + "=" * 70)
    print("第4轮：最终综合优化")
    print("=" * 70)

    results_final = []

    # 在最佳参数附近做更细的搜索
    vol_fine = [best_r1["vol_ratio"] - 0.5, best_r1["vol_ratio"], best_r1["vol_ratio"] + 0.5]
    vol_fine = [v for v in vol_fine if v >= 1.0]

    stop_min_configs = [0.008, 0.01, 0.012, 0.015]

    for vol_ratio, stop_min in product(vol_fine, stop_min_configs):
        # 测试开启和关闭时间止损
        for use_time in [True, False]:
            config = FOMOConfig(
                initial_capital=5000.0,
                risk_cash_per_campaign=50.0,
                atr_len=14,

                vol_ratio_entry=vol_ratio,
                vol_ratio_add=1.5,
                breakout_lookback=best_r1["breakout"],
                breakout_buffer_atr=best_r1["buffer"],

                k_stop_atr=best_r1["stop_atr"],
                stop_pct_min=stop_min,
                stop_pct_max=0.06,

                leverage_min=2,
                leverage_max=5,

                max_adds=3,
                add_levels_R=[1.5, 2.5, 3.5],
                add_notional_fracs=[0.6, 0.4, 0.3],
                require_stop_distance_atr=1.0,

                trail_k_atr=best_r2["trail_atr"],
                enable_trail_after_R=best_r2["trail_after"],

                time_stop_bars=best_time_bars,
                need_reach_R_in_time=best_need_r,

                daily_loss_limit=150.0,
                max_consecutive_losses=5,
                cooldown_bars=30,
                same_symbol_cooldown_bars=60,

                commission_rate=0.0006,
                slippage=0.001,
            )

            result = run_single_backtest(config, all_data, use_time_stop=use_time)

            results_final.append({
                "vol_ratio": vol_ratio,
                "breakout": best_r1["breakout"],
                "buffer": best_r1["buffer"],
                "stop_atr": best_r1["stop_atr"],
                "stop_min": stop_min,
                "trail_atr": best_r2["trail_atr"],
                "trail_after": best_r2["trail_after"],
                "time_bars": best_time_bars,
                "need_r": best_need_r,
                "use_time_stop": use_time,
                **result
            })

    results_final.sort(key=lambda x: x["total_pnl"], reverse=True)

    print(f"\n最终前15名:")
    print("-" * 140)
    print(f"{'排名':<4} {'Vol':>4} {'Brk':>4} {'Buf':>4} {'StpA':>5} {'StpM':>5} {'TrlA':>5} {'TrlR':>5} {'Time':>5} {'NdR':>4} {'TS':>3} | "
          f"{'PnL':>10} {'Trades':>6} {'WinR':>6} {'Exp(R)':>7}")
    print("-" * 140)

    for i, r in enumerate(results_final[:15], 1):
        ts_str = "ON" if r['use_time_stop'] else "OFF"
        print(f"{i:<4} {r['vol_ratio']:>4.1f} {r['breakout']:>4} {r['buffer']:>4.1f} {r['stop_atr']:>5.1f} "
              f"{r['stop_min']*100:>4.1f}% {r['trail_atr']:>5.1f} {r['trail_after']:>5.1f} {r['time_bars']:>5} {r['need_r']:>4.1f} {ts_str:>3} | "
              f"{r['total_pnl']:>+10.2f} {r['total_trades']:>6} {r['win_rate']:>5.1%} {r['expectancy']:>+7.2f}R")

    # 输出最佳配置
    best = results_final[0]

    print("\n" + "=" * 70)
    print("最佳参数配置")
    print("=" * 70)
    print(f"""
FOMOConfig(
    initial_capital=5000.0,
    risk_cash_per_campaign=50.0,

    atr_len=14,

    # 入场过滤
    vol_ratio_entry={best['vol_ratio']},
    vol_ratio_add=1.5,
    breakout_lookback={best['breakout']},
    breakout_buffer_atr={best['buffer']},

    # 止损
    k_stop_atr={best['stop_atr']},
    stop_pct_min={best['stop_min']},
    stop_pct_max=0.06,

    # 杠杆
    leverage_min=2,
    leverage_max=5,

    # 加仓
    max_adds=3,
    add_levels_R=[1.5, 2.5, 3.5],
    add_notional_fracs=[0.6, 0.4, 0.3],
    require_stop_distance_atr=1.0,

    # 追踪止损
    trail_k_atr={best['trail_atr']},
    enable_trail_after_R={best['trail_after']},

    # 时间止损 {'(开启)' if best['use_time_stop'] else '(关闭)'}
    time_stop_bars={best['time_bars']},
    need_reach_R_in_time={best['need_r']},

    # 风控
    daily_loss_limit=150.0,
    max_consecutive_losses=5,
    cooldown_bars=30,
    same_symbol_cooldown_bars=60,

    # 执行
    commission_rate=0.0006,
    slippage=0.001,
)

使用时间止损: {'是' if best['use_time_stop'] else '否'}
""")

    print(f"\n预期表现:")
    print(f"  总PnL:        {best['total_pnl']:+.2f} USDC")
    print(f"  总交易数:     {best['total_trades']}")
    print(f"  胜率:         {best['win_rate']:.1%}")
    print(f"  期望R:        {best['expectancy']:+.2f}R/交易")
    print(f"  平均盈利:     {best['avg_win_r']:+.2f}R")
    print(f"  平均亏损:     {best['avg_loss_r']:+.2f}R")
    print(f"  出场原因:     {best['exit_reasons']}")

    return best


if __name__ == "__main__":
    best_config = optimize_parameters()
