"""
Run Improved FOMO Strategy Backtest

Based on failure analysis findings:
1. 80.4% of trades exit via Time Stop - too aggressive
2. 0% of trades reach +2R - not letting winners run
3. Only 2.8% of trades have pyramid adds

Key improvements:
- Longer time window (60 bars instead of 20)
- Lower R threshold for time stop (0.3 instead of 0.5)
- Wider trailing stop (3.0 ATR instead of 2.0)
- Enable trailing later (1.5R instead of 0.8R)
- Higher entry bar (3.0x volume instead of 1.8x)
- Longer breakout lookback (40 instead of 20)
- Add EMA50 trend filter
"""
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from fetch_multi_pairs import MultiPairFetcher
from fomo_backtest import (
    FOMOConfig, FOMOBacktester, BacktestResult, Trade,
    Campaign, TechnicalIndicators
)


class ImprovedFOMOBacktester(FOMOBacktester):
    """
    Improved FOMO Backtester with:
    - EMA50 trend filter
    - ADX filter for trending markets
    - Better time management
    """

    def calculate_adx(self, highs: List[float], lows: List[float],
                      closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate ADX (Average Directional Index)"""
        if len(highs) < period * 2:
            return None

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]

            plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
            minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return None

        # Smoothed values
        def smooth(data, period):
            result = [sum(data[:period])]
            for i in range(period, len(data)):
                result.append(result[-1] - result[-1]/period + data[i])
            return result

        smooth_tr = smooth(tr_list, period)
        smooth_plus_dm = smooth(plus_dm, period)
        smooth_minus_dm = smooth(minus_dm, period)

        # DI values
        di_plus = []
        di_minus = []
        for i in range(len(smooth_tr)):
            if smooth_tr[i] > 0:
                di_plus.append(100 * smooth_plus_dm[i] / smooth_tr[i])
                di_minus.append(100 * smooth_minus_dm[i] / smooth_tr[i])
            else:
                di_plus.append(0)
                di_minus.append(0)

        # DX values
        dx_list = []
        for i in range(len(di_plus)):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx_list.append(100 * abs(di_plus[i] - di_minus[i]) / di_sum)
            else:
                dx_list.append(0)

        if len(dx_list) < period:
            return None

        # ADX is smoothed DX
        adx = np.mean(dx_list[-period:])
        return adx

    def run_backtest_improved(
        self,
        timestamps: List[datetime],
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        symbol: str = "BTC-USDC",
        use_trend_filter: bool = True,
        use_adx_filter: bool = True,
        adx_threshold: float = 20.0,
        verbose: bool = True
    ) -> BacktestResult:
        """Run improved FOMO strategy backtest with additional filters"""

        if verbose:
            print("=" * 70)
            print("IMPROVED FOMO STRATEGY BACKTEST")
            print("With Trend Filter + ADX Filter")
            print("=" * 70)
            print(f"Symbol: {symbol}")
            print(f"Data points: {len(closes)}")
            print(f"Period: {timestamps[0]} ~ {timestamps[-1]}")
            print(f"Trend Filter: {'ON' if use_trend_filter else 'OFF'}")
            print(f"ADX Filter: {'ON (>' + str(adx_threshold) + ')' if use_adx_filter else 'OFF'}")
            print("=" * 70)

        # State tracking
        capital = self.config.initial_capital
        campaign: Optional[Campaign] = None
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []

        # Global risk state
        daily_pnl = 0.0
        consecutive_losses = 0
        cooldown_until = 0
        symbol_cooldown: Dict[str, int] = {}

        # Stats
        peak_equity = capital
        max_drawdown = 0
        max_drawdown_pct = 0
        max_consec_losses_seen = 0

        # Tracking filters
        trades_blocked_trend = 0
        trades_blocked_adx = 0
        entries_attempted = 0

        min_bars = max(self.config.atr_len + 10, self.config.breakout_lookback + 10, 70)

        for i in range(min_bars, len(closes)):
            current_time = timestamps[i]
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Check for new day
            if i > 0 and timestamps[i].date() != timestamps[i-1].date():
                daily_pnl = 0.0

            # Calculate ATR
            atr = self.indicators.calculate_atr(
                highs[:i+1], lows[:i+1], closes[:i+1], self.config.atr_len
            )
            if not atr:
                continue

            # Calculate EMA50 for trend filter
            ema50 = self.indicators.calculate_ema(closes[:i+1], 50)

            # Calculate EMA9 for confirmation
            ema9 = self.indicators.calculate_ema(closes[:i+1], 9)

            # Calculate ADX for market regime
            adx = self.calculate_adx(highs[:i+1], lows[:i+1], closes[:i+1], 14)

            # Current equity
            if campaign:
                unrealized_pnl = (current_price - campaign.avg_entry) * campaign.total_qty
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity_curve.append((current_time, current_equity))

            # Drawdown tracking
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = peak_equity - current_equity
            drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

            # ========== POSITION MANAGEMENT ==========
            if campaign:
                should_exit = False
                exit_reason = ""
                exit_price = current_price

                # Update highest
                if current_high > campaign.highest_since_entry:
                    campaign.highest_since_entry = current_high

                # Update trailing stop
                if campaign.trail_enabled:
                    new_trail = campaign.highest_since_entry - self.config.trail_k_atr * atr
                    if campaign.trail_stop is None or new_trail > campaign.trail_stop:
                        campaign.trail_stop = new_trail

                # Enable trailing after R threshold
                if not campaign.trail_enabled:
                    profit_R = (current_price - campaign.initial_entry_price) / campaign.R_price
                    if profit_R >= self.config.enable_trail_after_R:
                        campaign.trail_enabled = True
                        campaign.trail_stop = campaign.highest_since_entry - self.config.trail_k_atr * atr

                # Effective stop
                effective_stop = campaign.current_stop
                if campaign.trail_stop and campaign.trail_stop > effective_stop:
                    effective_stop = campaign.trail_stop

                # Check stops
                if current_low <= effective_stop:
                    should_exit = True
                    exit_price = effective_stop
                    if campaign.trail_stop and effective_stop == campaign.trail_stop:
                        exit_reason = f"Trailing Stop @ {effective_stop:.2f}"
                    else:
                        exit_reason = f"Stop Loss @ {effective_stop:.2f}"

                # Time stop
                if not should_exit:
                    bars_held = i - campaign.entry_bar_idx
                    if bars_held >= self.config.time_stop_bars:
                        profit_R = (current_price - campaign.initial_entry_price) / campaign.R_price
                        if profit_R < self.config.need_reach_R_in_time:
                            should_exit = True
                            exit_reason = f"Time Stop ({bars_held} bars, +{profit_R:.2f}R)"

                # Check adding
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

                            if verbose:
                                R_profit = (current_price - campaign.initial_entry_price) / campaign.R_price
                                print(f"  [{current_time}] ADD #{campaign.add_count} @ {add_price:.2f} | +{R_profit:.2f}R")

                # Execute exit
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

                    if verbose:
                        result_str = "WIN" if pnl > 0 else "LOSS"
                        print(f"[{current_time}] CLOSE @ {exit_price:.2f} | PnL: {pnl:+.2f} ({r_multiple:+.2f}R) | {exit_reason} | {result_str}")

                    campaign = None

            # ========== NEW ENTRY CHECK ==========
            if campaign is None:
                # Global risk limits
                if daily_pnl <= -self.config.daily_loss_limit:
                    continue
                if i < cooldown_until:
                    continue
                if symbol in symbol_cooldown and i < symbol_cooldown[symbol]:
                    continue

                # Check breakout
                is_breakout, breakout_level = self.check_breakout(highs, closes, i, atr)
                if not is_breakout:
                    continue

                # Volume confirmation
                vol_confirmed, vol_ratio = self.check_volume_confirmation(
                    volumes, i, 60, self.config.vol_ratio_entry
                )
                if not vol_confirmed:
                    continue

                entries_attempted += 1

                # ===== NEW FILTERS =====

                # Trend filter: Price must be above EMA50
                if use_trend_filter and ema50:
                    if current_price < ema50:
                        trades_blocked_trend += 1
                        continue

                # ADX filter: Market must be trending
                if use_adx_filter and adx:
                    if adx < adx_threshold:
                        trades_blocked_adx += 1
                        continue

                # ===== END NEW FILTERS =====

                # Calculate stop
                stop_pct, is_valid = self.calculate_stop_pct(atr, current_price)
                if not is_valid:
                    continue

                # Position size
                notional, qty = self.calculate_initial_position(current_price, stop_pct)

                # Margin check
                leverage = self.calculate_leverage(stop_pct)
                required_margin = notional / leverage
                if required_margin > capital * 0.5:
                    continue

                # Entry
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

                if verbose:
                    adx_str = f" | ADX: {adx:.1f}" if adx else ""
                    print(f"[{current_time}] OPEN LONG @ {entry_price:.2f} | Stop: {stop_price:.2f} ({stop_pct*100:.2f}%) | Vol: {vol_ratio:.1f}x{adx_str}")

        # Close remaining
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

        # Statistics
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

        if verbose:
            print(f"\nFilter Stats:")
            print(f"  Entry attempts: {entries_attempted}")
            print(f"  Blocked by trend filter: {trades_blocked_trend}")
            print(f"  Blocked by ADX filter: {trades_blocked_adx}")
            print(f"  Actual entries: {len(trades)}")

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


def run_improved_backtest(
    days: int = 3,
    interval: str = "1m",
    use_trend_filter: bool = True,
    use_adx_filter: bool = True,
    adx_threshold: float = 20.0
) -> Dict[str, BacktestResult]:
    """Run improved FOMO backtest on multiple pairs"""

    print("=" * 70)
    print("IMPROVED FOMO STRATEGY - MULTI-PAIR BACKTEST")
    print("=" * 70)

    fetcher = MultiPairFetcher()

    # Get hot pairs
    print("\n[1] Scanning for hot pairs...")
    hot_pairs = fetcher.get_hot_pairs(top_n=10)
    pairs = [p.get('symbol') for p in hot_pairs]

    print(f"\nSelected {len(pairs)} pairs:")
    for i, p in enumerate(hot_pairs):
        print(f"  {i+1}. {p.get('symbol'):<12} Change: {p.get('priceChangeFloat', 0):+.2f}%")

    # Fetch data
    print(f"\n[2] Fetching {interval} data for {len(pairs)} pairs ({days} days)...")
    all_data = fetcher.fetch_all_pairs_data(pairs, interval, days)

    if not all_data:
        print("ERROR: No data fetched")
        return {}

    # IMPROVED CONFIG based on analysis
    config = FOMOConfig(
        initial_capital=5000.0,
        risk_cash_per_campaign=50.0,

        atr_len=14,

        # Entry - MORE SELECTIVE
        vol_ratio_entry=3.0,          # Was 1.8 - require stronger volume
        vol_ratio_add=2.0,            # Was 1.3
        breakout_lookback=40,         # Was 20 - longer lookback
        breakout_buffer_atr=0.4,      # Was 0.25 - more buffer

        # Stop - GIVE MORE ROOM
        k_stop_atr=2.5,               # Was 1.8 - wider stops
        stop_pct_min=0.012,           # Was 0.008 - 1.2% min
        stop_pct_max=0.06,

        # Leverage
        leverage_min=2,
        leverage_max=5,

        # Pyramid - HIGHER BAR
        max_adds=3,
        add_levels_R=[1.5, 2.5, 3.5],  # Was [1.0, 2.0, 3.0]
        add_notional_fracs=[0.6, 0.4, 0.3],
        require_stop_distance_atr=1.0,

        # Exit - LET WINNERS RUN
        trail_k_atr=3.0,              # Was 2.0 - wider trailing
        enable_trail_after_R=1.5,     # Was 0.8 - wait longer

        # Time - BE PATIENT
        time_stop_bars=60,            # Was 20 - 1 hour instead of 20 min
        need_reach_R_in_time=0.3,     # Was 0.5 - lower threshold

        # Global risk
        daily_loss_limit=150.0,
        max_consecutive_losses=5,
        cooldown_bars=30,
        same_symbol_cooldown_bars=60,

        # Execution
        commission_rate=0.0006,
        slippage=0.001,
    )

    print(f"\n[3] Improved Config:")
    print(f"    Entry Filter:    Vol >= {config.vol_ratio_entry}x + Breakout({config.breakout_lookback})")
    print(f"    Stop Range:      {config.stop_pct_min*100:.1f}% - {config.stop_pct_max*100:.1f}%")
    print(f"    Time Stop:       {config.time_stop_bars} bars (need {config.need_reach_R_in_time}R)")
    print(f"    Trend Filter:    {'ON' if use_trend_filter else 'OFF'}")
    print(f"    ADX Filter:      {'>' + str(adx_threshold) if use_adx_filter else 'OFF'}")

    # Run backtest
    print(f"\n[4] Running improved backtests...")
    print("=" * 70)

    results = {}
    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for symbol, data in all_data.items():
        timestamps, opens, highs, lows, closes, volumes = data

        if len(timestamps) < 100:
            print(f"  {symbol}: Skipped (not enough data)")
            continue

        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print(f"{'='*70}")

        backtester = ImprovedFOMOBacktester(config)

        try:
            result = backtester.run_backtest_improved(
                timestamps=timestamps,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                symbol=symbol,
                use_trend_filter=use_trend_filter,
                use_adx_filter=use_adx_filter,
                adx_threshold=adx_threshold,
                verbose=True
            )

            results[symbol] = result
            total_pnl += result.total_return
            total_trades += result.total_trades
            total_wins += result.winning_trades

            print(f"\n  Result: {result.total_return:+.2f} USDC ({result.total_return_pct:+.2%})")
            print(f"  Trades: {result.total_trades} | Win Rate: {result.win_rate:.2%}")
            print(f"  Expectancy: {result.expectancy_R:+.2f}R | Max DD: {result.max_drawdown_pct:.2%}")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("IMPROVED STRATEGY - OVERALL SUMMARY")
    print("=" * 70)

    print(f"\n{'Symbol':<12} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Return':>10} {'Exp(R)':>10} {'MaxDD':>10}")
    print("-" * 70)

    for symbol, result in sorted(results.items(), key=lambda x: x[1].total_return, reverse=True):
        print(f"{symbol:<12} {result.total_trades:>8} {result.win_rate:>7.1%} "
              f"{result.total_return:>+12.2f} {result.total_return_pct:>+9.2%} "
              f"{result.expectancy_R:>+9.2f}R {result.max_drawdown_pct:>9.2%}")

    print("-" * 70)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    print(f"{'TOTAL':<12} {total_trades:>8} {overall_wr:>7.1%} {total_pnl:>+12.2f}")

    # Compare to original
    if results:
        profitable = [s for s, r in results.items() if r.total_return > 0]
        print(f"\nProfitable pairs: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.0f}%)")

        # Exit reason analysis
        all_trades = []
        for r in results.values():
            all_trades.extend(r.trades)

        if all_trades:
            exit_reasons = {}
            for t in all_trades:
                reason = t.exit_reason.split("@")[0].strip() if "@" in t.exit_reason else t.exit_reason
                if reason not in exit_reasons:
                    exit_reasons[reason] = {"count": 0, "pnl": 0}
                exit_reasons[reason]["count"] += 1
                exit_reasons[reason]["pnl"] += t.pnl

            print(f"\nExit Reason Breakdown:")
            for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
                pct = stats["count"] / len(all_trades) * 100
                avg_pnl = stats["pnl"] / stats["count"]
                print(f"  {reason:<25}: {stats['count']:>4} ({pct:>5.1f}%) | Avg PnL: {avg_pnl:>+8.2f}")

    return results


if __name__ == "__main__":
    days = 3
    interval = "1m"

    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    if len(sys.argv) > 2:
        interval = sys.argv[2]

    # Run with improvements
    results = run_improved_backtest(
        days=days,
        interval=interval,
        use_trend_filter=True,
        use_adx_filter=True,
        adx_threshold=20.0
    )
