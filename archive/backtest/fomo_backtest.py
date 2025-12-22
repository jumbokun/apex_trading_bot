"""
FOMO Strategy Backtest - Hot Trend Momentum + Risk-Capped Pyramid Adding

Core principles:
1. Entry: Only enter when "truly surging" (breakout + volume confirmation)
2. Adding: Only add when getting stronger (anti-martingale),
   each add raises stop-loss to keep max loss <= 50U
3. Exit: ATR-driven trailing stop + time stop

Author: Apex Trading Bot
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import numpy as np


@dataclass
class FOMOConfig:
    """FOMO Strategy Configuration"""

    # Core risk parameter - THE fundamental constraint
    risk_cash_per_campaign: float = 50.0  # Max loss per trade campaign = 50U
    initial_capital: float = 5000.0

    # ATR settings
    atr_len: int = 14

    # Hot trend filter
    top_k_ret_15m: int = 5  # Only trade top K gainers
    vol_ratio_entry: float = 2.0  # Volume must be 2x average to enter
    vol_ratio_add: float = 1.3  # Volume must be 1.3x to add position

    # Entry - Breakout detection
    breakout_lookback: int = 20  # Look at last 20 candles for high
    breakout_buffer_atr: float = 0.2  # Buffer = 0.2 * ATR

    # Stop loss - ATR based with clamping
    k_stop_atr: float = 1.6  # stop_dist = 1.6 * ATR
    stop_pct_min: float = 0.01  # Min stop = 1%
    stop_pct_max: float = 0.06  # Max stop = 6% (skip if higher)

    # Leverage - derived from stop distance
    leverage_min: int = 2
    leverage_max: int = 5

    # Pyramid adding
    max_adds: int = 3
    add_levels_R: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    add_notional_fracs: List[float] = field(default_factory=lambda: [0.6, 0.4, 0.3])
    max_total_notional_mult: float = 2.5
    require_stop_distance_atr: float = 1.0  # Min distance from stop after adding

    # Exit - Trailing stop
    trail_k_atr: float = 3.0  # trail_stop = highest - 3*ATR
    enable_trail_after_R: float = 1.0  # Enable trailing after +1R profit

    # Exit - Time stop
    time_stop_bars: int = 12  # Exit if no progress in 12 bars (12 min for 1m)
    need_reach_R_in_time: float = 0.8  # Must reach +0.8R within time window

    # Global risk controls
    daily_loss_limit: float = 150.0  # Max daily loss = 3R
    max_consecutive_losses: int = 3
    cooldown_bars: int = 60  # Cooldown after consecutive losses
    same_symbol_cooldown_bars: int = 30  # Cooldown for same symbol after stop
    max_open_positions: int = 1

    # Execution
    commission_rate: float = 0.0006
    slippage: float = 0.001
    max_slippage: float = 0.005  # Skip if slippage > 0.5%


@dataclass
class Campaign:
    """
    A trading campaign (can include multiple adds)
    This is the core unit - max loss for entire campaign = risk_cash
    """
    symbol: str
    side: str  # "LONG" only for FOMO (we chase uptrends)

    # Position tracking
    entries: List[Tuple[datetime, float, float]] = field(default_factory=list)  # [(time, price, size)]
    total_qty: float = 0.0
    avg_entry: float = 0.0

    # Initial parameters (set at first entry)
    initial_entry_price: float = 0.0
    initial_stop: float = 0.0
    R_price: float = 0.0  # 1R = initial_entry - initial_stop

    # Current stop (can only go UP)
    current_stop: float = 0.0

    # Trailing stop tracking
    highest_since_entry: float = 0.0
    trail_stop: Optional[float] = None
    trail_enabled: bool = False

    # Adding tracking
    add_count: int = 0
    total_notional: float = 0.0

    # Time tracking
    entry_bar_idx: int = 0

    # ATR at entry (for trailing)
    entry_atr: float = 0.0


@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    avg_entry_price: float
    exit_price: float
    total_qty: float
    add_count: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    r_multiple: float  # How many R did we make/lose


@dataclass
class BacktestResult:
    """Backtest result summary"""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    total_profit: float
    total_loss: float
    profit_factor: float

    avg_win_R: float
    avg_loss_R: float
    expectancy_R: float  # Expected R per trade

    max_drawdown: float
    max_drawdown_pct: float
    max_consecutive_losses: int

    avg_adds_per_trade: float

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class TechnicalIndicators:
    """Technical indicator calculations"""

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        return np.mean(true_ranges[-period:])

    @staticmethod
    def calculate_ema(data: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def highest(data: List[float], period: int) -> float:
        """Get highest value in last N periods"""
        return max(data[-period:]) if len(data) >= period else max(data)

    @staticmethod
    def lowest(data: List[float], period: int) -> float:
        """Get lowest value in last N periods"""
        return min(data[-period:]) if len(data) >= period else min(data)


class FOMOBacktester:
    """
    FOMO Strategy Backtester

    Core innovation: Risk-capped pyramid adding
    - Each campaign has fixed max loss = 50U
    - Every time we add, we RAISE the stop to maintain max loss
    """

    def __init__(self, config: FOMOConfig):
        self.config = config
        self.indicators = TechnicalIndicators()

    def calculate_stop_pct(self, atr: float, price: float) -> Tuple[float, bool]:
        """
        Calculate stop loss percentage based on ATR
        Returns (stop_pct, is_valid)
        """
        stop_dist = self.config.k_stop_atr * atr
        stop_pct = stop_dist / price

        # Clamp to min/max
        if stop_pct < self.config.stop_pct_min:
            stop_pct = self.config.stop_pct_min
        elif stop_pct > self.config.stop_pct_max:
            # Skip this trade - too volatile
            return stop_pct, False

        return stop_pct, True

    def calculate_leverage(self, stop_pct: float) -> int:
        """
        Calculate leverage based on stop distance
        Rule: Liquidation distance >= 4x stop distance
        """
        L_cap = int(1 / (4 * stop_pct))
        L = max(self.config.leverage_min, min(L_cap, self.config.leverage_max))
        return L

    def calculate_initial_position(self, price: float, stop_pct: float) -> Tuple[float, float]:
        """
        Calculate initial position size based on fixed risk
        notional = risk_cash / stop_pct

        Returns (notional_value, qty)
        """
        notional = self.config.risk_cash_per_campaign / stop_pct
        qty = notional / price
        return notional, qty

    def calculate_risk_stop(self, avg_entry: float, total_qty: float) -> float:
        """
        Calculate the stop price that limits loss to risk_cash
        This is THE core formula for risk-capped adding

        risk_stop = avg_entry - (risk_cash / total_qty)
        """
        if total_qty <= 0:
            return 0
        return avg_entry - (self.config.risk_cash_per_campaign / total_qty)

    def check_volume_confirmation(self, volumes: List[float], current_idx: int,
                                   ma_period: int = 60, required_ratio: float = 2.0) -> Tuple[bool, float]:
        """
        Check if current volume is sufficient
        Returns (is_confirmed, volume_ratio)
        """
        if current_idx < ma_period:
            return False, 0

        vol_ma = np.mean(volumes[current_idx - ma_period:current_idx])
        current_vol = volumes[current_idx]
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 0

        return vol_ratio >= required_ratio, vol_ratio

    def check_breakout(self, highs: List[float], closes: List[float],
                       current_idx: int, atr: float) -> Tuple[bool, float]:
        """
        Check for breakout condition
        close > highest(high, N) + buffer

        Returns (is_breakout, breakout_level)
        """
        lookback = self.config.breakout_lookback
        if current_idx < lookback:
            return False, 0

        # Get highest high in lookback period (excluding current bar)
        recent_highs = highs[current_idx - lookback:current_idx]
        highest_high = max(recent_highs)

        # Add buffer
        buffer = self.config.breakout_buffer_atr * atr
        breakout_level = highest_high + buffer

        # Check if current close breaks above
        current_close = closes[current_idx]
        is_breakout = current_close > breakout_level

        return is_breakout, breakout_level

    def should_add_position(self, campaign: Campaign, current_price: float,
                           atr: float, vol_ratio: float, ema9: float) -> bool:
        """
        Check if we should add to position
        Conditions:
        1. Price reached next R level
        2. Volume still confirming
        3. Price above EMA9
        4. Stop distance after add >= 1 ATR
        """
        if campaign.add_count >= self.config.max_adds:
            return False

        # Check R level
        next_add_idx = campaign.add_count
        if next_add_idx >= len(self.config.add_levels_R):
            return False

        required_R = self.config.add_levels_R[next_add_idx]
        target_price = campaign.initial_entry_price + required_R * campaign.R_price

        if current_price < target_price:
            return False

        # Check volume
        if vol_ratio < self.config.vol_ratio_add:
            return False

        # Check trend (price above EMA9)
        if ema9 and current_price < ema9:
            return False

        # Check that adding won't put stop too close
        # Calculate what the new stop would be after adding
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
        """Run the FOMO strategy backtest"""

        print("=" * 70)
        print("FOMO STRATEGY BACKTEST")
        print("Hot Trend Momentum + Risk-Capped Pyramid Adding")
        print("=" * 70)
        print(f"Symbol: {symbol}")
        print(f"Data points: {len(closes)}")
        print(f"Period: {timestamps[0]} ~ {timestamps[-1]}")
        print(f"Initial Capital: {self.config.initial_capital:.2f} USDC")
        print(f"Risk per campaign: {self.config.risk_cash_per_campaign:.2f} USDC")
        print("=" * 70)

        # State tracking
        capital = self.config.initial_capital
        campaign: Optional[Campaign] = None
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []

        # Global risk state
        daily_pnl = 0.0
        daily_start_bar = 0
        consecutive_losses = 0
        cooldown_until = 0
        symbol_cooldown: Dict[str, int] = {}

        # Stats
        peak_equity = capital
        max_drawdown = 0
        max_drawdown_pct = 0
        max_consec_losses_seen = 0

        # Minimum data requirement
        min_bars = max(self.config.atr_len + 10, self.config.breakout_lookback + 10, 70)

        for i in range(min_bars, len(closes)):
            current_time = timestamps[i]
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Check for new day (reset daily PnL)
            if i > 0 and timestamps[i].date() != timestamps[i-1].date():
                daily_pnl = 0.0
                daily_start_bar = i

            # Calculate current ATR
            atr = self.indicators.calculate_atr(
                highs[:i+1], lows[:i+1], closes[:i+1], self.config.atr_len
            )
            if not atr:
                continue

            # Calculate EMA9 for trend confirmation
            ema9 = self.indicators.calculate_ema(closes[:i+1], 9)

            # Calculate current equity
            if campaign:
                unrealized_pnl = (current_price - campaign.avg_entry) * campaign.total_qty
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity_curve.append((current_time, current_equity))

            # Update max drawdown
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

                # Update highest since entry
                if current_high > campaign.highest_since_entry:
                    campaign.highest_since_entry = current_high

                # Update trailing stop if enabled
                if campaign.trail_enabled:
                    new_trail = campaign.highest_since_entry - self.config.trail_k_atr * atr
                    if campaign.trail_stop is None or new_trail > campaign.trail_stop:
                        campaign.trail_stop = new_trail

                # Enable trailing stop after +1R
                if not campaign.trail_enabled:
                    profit_R = (current_price - campaign.initial_entry_price) / campaign.R_price
                    if profit_R >= self.config.enable_trail_after_R:
                        campaign.trail_enabled = True
                        campaign.trail_stop = campaign.highest_since_entry - self.config.trail_k_atr * atr

                # Check stops (in priority order)
                effective_stop = campaign.current_stop
                if campaign.trail_stop and campaign.trail_stop > effective_stop:
                    effective_stop = campaign.trail_stop

                # 1. Hard stop hit
                if current_low <= effective_stop:
                    should_exit = True
                    exit_price = effective_stop
                    if campaign.trail_stop and effective_stop == campaign.trail_stop:
                        exit_reason = f"Trailing Stop (ATR) @ {effective_stop:.2f}"
                    else:
                        exit_reason = f"Stop Loss @ {effective_stop:.2f}"

                # 2. Time stop - no progress within window
                if not should_exit:
                    bars_held = i - campaign.entry_bar_idx
                    if bars_held >= self.config.time_stop_bars:
                        profit_R = (current_price - campaign.initial_entry_price) / campaign.R_price
                        if profit_R < self.config.need_reach_R_in_time:
                            should_exit = True
                            exit_reason = f"Time Stop ({bars_held} bars, only +{profit_R:.2f}R)"

                # 3. Check for adding position
                if not should_exit:
                    vol_confirmed, vol_ratio = self.check_volume_confirmation(
                        volumes, i, 60, self.config.vol_ratio_add
                    )

                    if self.should_add_position(campaign, current_price, atr, vol_ratio, ema9):
                        # Calculate add size
                        add_idx = campaign.add_count
                        add_frac = self.config.add_notional_fracs[add_idx]

                        # Initial notional was based on first entry
                        initial_notional = campaign.entries[0][1] * campaign.entries[0][2]
                        add_notional = initial_notional * add_frac

                        # Check total notional limit
                        potential_total = campaign.total_notional + add_notional
                        max_allowed = initial_notional * self.config.max_total_notional_mult

                        if potential_total <= max_allowed:
                            # Apply slippage
                            add_price = current_price * (1 + self.config.slippage)
                            add_qty = add_notional / add_price

                            # Deduct commission
                            commission = add_notional * self.config.commission_rate
                            capital -= commission

                            # Update campaign
                            new_total_qty = campaign.total_qty + add_qty
                            new_avg = (campaign.avg_entry * campaign.total_qty + add_price * add_qty) / new_total_qty

                            campaign.entries.append((current_time, add_price, add_qty))
                            campaign.total_qty = new_total_qty
                            campaign.avg_entry = new_avg
                            campaign.total_notional += add_notional
                            campaign.add_count += 1

                            # CRITICAL: Raise stop to maintain max risk
                            new_risk_stop = self.calculate_risk_stop(new_avg, new_total_qty)
                            campaign.current_stop = max(campaign.current_stop, new_risk_stop)

                            R_profit = (current_price - campaign.initial_entry_price) / campaign.R_price
                            print(f"  [{current_time}] ADD #{campaign.add_count} @ {add_price:.2f} | "
                                  f"Qty: +{add_qty:.6f} | Total: {campaign.total_qty:.6f} | "
                                  f"New Stop: {campaign.current_stop:.2f} | +{R_profit:.2f}R")

                # Execute exit if needed
                if should_exit:
                    # Apply slippage on exit
                    exit_price = exit_price * (1 - self.config.slippage)
                    pnl = (exit_price - campaign.avg_entry) * campaign.total_qty

                    # Deduct commission
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

                    # Update consecutive losses
                    if pnl < 0:
                        consecutive_losses += 1
                        if consecutive_losses > max_consec_losses_seen:
                            max_consec_losses_seen = consecutive_losses

                        # Set cooldown if needed
                        if consecutive_losses >= self.config.max_consecutive_losses:
                            cooldown_until = i + self.config.cooldown_bars

                        # Set symbol cooldown
                        symbol_cooldown[symbol] = i + self.config.same_symbol_cooldown_bars
                    else:
                        consecutive_losses = 0

                    result_str = "WIN" if pnl > 0 else "LOSS"
                    print(f"[{current_time}] CLOSE {campaign.side} @ {exit_price:.2f} | "
                          f"Adds: {campaign.add_count} | PnL: {pnl:+.2f} ({r_multiple:+.2f}R) | "
                          f"{exit_reason} | {result_str}")

                    campaign = None

            # ========== NEW ENTRY CHECK ==========
            if campaign is None:
                # Check global risk limits
                if daily_pnl <= -self.config.daily_loss_limit:
                    continue  # Daily limit hit

                if i < cooldown_until:
                    continue  # In cooldown

                if symbol in symbol_cooldown and i < symbol_cooldown[symbol]:
                    continue  # Symbol cooldown

                # Check for breakout
                is_breakout, breakout_level = self.check_breakout(highs, closes, i, atr)
                if not is_breakout:
                    continue

                # Check volume confirmation
                vol_confirmed, vol_ratio = self.check_volume_confirmation(
                    volumes, i, 60, self.config.vol_ratio_entry
                )
                if not vol_confirmed:
                    continue

                # Calculate stop and check if trade is valid
                stop_pct, is_valid = self.calculate_stop_pct(atr, current_price)
                if not is_valid:
                    continue  # Too volatile

                # Calculate position size based on fixed risk
                notional, qty = self.calculate_initial_position(current_price, stop_pct)

                # Check if we have enough capital
                leverage = self.calculate_leverage(stop_pct)
                required_margin = notional / leverage
                if required_margin > capital * 0.5:
                    continue  # Not enough margin

                # Entry with slippage
                entry_price = current_price * (1 + self.config.slippage)
                qty = notional / entry_price

                # Calculate stop loss
                stop_price = entry_price * (1 - stop_pct)
                R_price = entry_price - stop_price

                # Deduct commission
                commission = notional * self.config.commission_rate
                capital -= commission

                # Create campaign
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

                print(f"[{current_time}] OPEN LONG @ {entry_price:.2f} | "
                      f"Qty: {qty:.6f} | Stop: {stop_price:.2f} ({stop_pct*100:.2f}%) | "
                      f"1R = {R_price:.2f} | Vol: {vol_ratio:.1f}x | L: {leverage}x")

        # Close remaining position at end
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

            print(f"[{timestamps[-1]}] End of Backtest CLOSE @ {final_price:.2f} | "
                  f"PnL: {pnl:+.2f} ({r_multiple:+.2f}R)")

        # Calculate statistics
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

    def print_result(self, result: BacktestResult):
        """Print formatted backtest results"""
        print("\n" + "=" * 70)
        print("FOMO STRATEGY BACKTEST RESULTS")
        print("=" * 70)

        print(f"\n[Capital]")
        print(f"  Initial:    {result.initial_capital:>12,.2f} USDC")
        print(f"  Final:      {result.final_capital:>12,.2f} USDC")
        print(f"  Return:     {result.total_return:>+12,.2f} USDC ({result.total_return_pct:+.2%})")

        print(f"\n[Trade Statistics]")
        print(f"  Total Trades:       {result.total_trades:>8}")
        print(f"  Winning:            {result.winning_trades:>8}")
        print(f"  Losing:             {result.losing_trades:>8}")
        print(f"  Win Rate:           {result.win_rate:>8.2%}")
        print(f"  Avg Adds/Trade:     {result.avg_adds_per_trade:>8.2f}")

        print(f"\n[R-Multiple Analysis] (1R = {self.config.risk_cash_per_campaign}U risk)")
        print(f"  Avg Win:            {result.avg_win_R:>+8.2f}R")
        print(f"  Avg Loss:           {result.avg_loss_R:>+8.2f}R")
        print(f"  Expectancy:         {result.expectancy_R:>+8.2f}R per trade")

        print(f"\n[Profit Analysis]")
        print(f"  Total Profit:       {result.total_profit:>12,.2f} USDC")
        print(f"  Total Loss:         {result.total_loss:>12,.2f} USDC")
        print(f"  Profit Factor:      {result.profit_factor:>12.2f}")

        print(f"\n[Risk Metrics]")
        print(f"  Max Drawdown:       {result.max_drawdown:>12,.2f} USDC ({result.max_drawdown_pct:.2%})")
        print(f"  Max Consec Losses:  {result.max_consecutive_losses:>12}")

        # Strategy evaluation
        print(f"\n[Strategy Evaluation]")

        if result.expectancy_R > 0:
            print(f"  [+] Positive expectancy: {result.expectancy_R:+.2f}R per trade")
        else:
            print(f"  [-] Negative expectancy: {result.expectancy_R:+.2f}R per trade")

        if result.profit_factor >= 1.5:
            print(f"  [+] Excellent profit factor: {result.profit_factor:.2f}")
        elif result.profit_factor >= 1.0:
            print(f"  [=] Positive profit factor: {result.profit_factor:.2f}")
        else:
            print(f"  [-] Poor profit factor: {result.profit_factor:.2f}")

        if result.win_rate >= 0.4:
            print(f"  [+] Good win rate for trend following: {result.win_rate:.2%}")
        else:
            print(f"  [!] Low win rate: {result.win_rate:.2%} (check if avg win >> avg loss)")

        if result.max_drawdown_pct <= 0.10:
            print(f"  [+] Excellent drawdown control: {result.max_drawdown_pct:.2%}")
        elif result.max_drawdown_pct <= 0.20:
            print(f"  [=] Acceptable drawdown: {result.max_drawdown_pct:.2%}")
        else:
            print(f"  [-] Large drawdown: {result.max_drawdown_pct:.2%}")

        # Risk-reward analysis
        if result.avg_win_R and result.avg_loss_R:
            rr_ratio = abs(result.avg_win_R / result.avg_loss_R) if result.avg_loss_R else 0
            print(f"  [i] Risk/Reward Ratio: 1:{rr_ratio:.2f}")

            # Calculate required win rate for breakeven
            if rr_ratio > 0:
                breakeven_wr = 1 / (1 + rr_ratio)
                print(f"  [i] Breakeven Win Rate: {breakeven_wr:.2%}")

        print("\n" + "=" * 70)

        # Show recent trades
        if result.trades:
            print("\nRecent 10 Trades:")
            print("-" * 100)
            print(f"{'Entry Time':<18} | {'Exit Time':<18} | {'Adds':>4} | "
                  f"{'Entry':>10} | {'Exit':>10} | {'PnL':>10} | {'R-Mult':>7} | Exit Reason")
            print("-" * 100)
            for trade in result.trades[-10:]:
                print(f"{trade.entry_time.strftime('%m-%d %H:%M'):<18} | "
                      f"{trade.exit_time.strftime('%m-%d %H:%M'):<18} | "
                      f"{trade.add_count:>4} | "
                      f"{trade.avg_entry_price:>10.2f} | "
                      f"{trade.exit_price:>10.2f} | "
                      f"{trade.pnl:>+10.2f} | "
                      f"{trade.r_multiple:>+7.2f}R | "
                      f"{trade.exit_reason}")


def generate_trending_data(days: int = 30, interval_minutes: int = 1) -> Tuple[List, List, List, List, List, List]:
    """
    Generate sample data with trending behavior suitable for FOMO strategy testing
    Includes:
    - Trend phases
    - Breakout patterns
    - Volume spikes during breakouts
    """
    print("Generating trending sample data...")

    np.random.seed(42)

    points_per_day = 24 * 60 // interval_minutes
    total_points = days * points_per_day

    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=interval_minutes * i) for i in range(total_points)]

    # Base price
    base_price = 95000
    prices = [base_price]
    volumes = []
    base_volume = 1000

    # Generate price with trending behavior
    for i in range(1, total_points):
        # Create trend phases
        phase = (i // (points_per_day * 3)) % 4  # 3-day phases

        if phase == 0:  # Uptrend
            drift = 0.0002
            volatility = 0.008
        elif phase == 1:  # Consolidation
            drift = 0.00005
            volatility = 0.005
        elif phase == 2:  # Breakout
            drift = 0.0004
            volatility = 0.012
        else:  # Pullback
            drift = -0.0001
            volatility = 0.01

        # Random component
        random_return = np.random.normal(drift, volatility)

        # Add some mean reversion
        if i > 100:
            ma = np.mean(prices[-100:])
            mean_reversion = (ma - prices[-1]) / prices[-1] * 0.01
            random_return += mean_reversion

        new_price = prices[-1] * (1 + random_return)
        prices.append(new_price)

        # Volume - spike during big moves
        price_change = abs(random_return)
        volume_mult = 1 + (price_change / volatility) * 2
        volume_noise = np.random.uniform(0.7, 1.3)
        volumes.append(base_volume * volume_mult * volume_noise)

    volumes.insert(0, base_volume)

    # Generate OHLC
    opens = []
    highs = []
    lows = []
    closes = []

    for i, close in enumerate(prices):
        if i == 0:
            open_price = base_price
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.001))

        # Extend high/low
        range_pct = abs(np.random.normal(0.003, 0.002))
        high = max(open_price, close) * (1 + range_pct)
        low = min(open_price, close) * (1 - range_pct)

        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        closes.append(close)

    print(f"Generated {total_points} data points ({days} days, {interval_minutes}m interval)")
    print(f"Price range: {min(closes):.2f} - {max(closes):.2f}")

    return timestamps, opens, highs, lows, closes, volumes


def run_fomo_backtest_demo():
    """Run FOMO strategy backtest demo"""
    print("=" * 70)
    print("FOMO STRATEGY - Hot Trend Momentum + Risk-Capped Pyramid")
    print("=" * 70)

    # Create config
    config = FOMOConfig(
        initial_capital=5000.0,
        risk_cash_per_campaign=50.0,  # Max loss per campaign = 50U

        # ATR
        atr_len=14,

        # Entry
        vol_ratio_entry=2.0,
        breakout_lookback=20,
        breakout_buffer_atr=0.2,

        # Stop
        k_stop_atr=1.6,
        stop_pct_min=0.01,
        stop_pct_max=0.06,

        # Leverage
        leverage_min=2,
        leverage_max=5,

        # Pyramid
        max_adds=3,
        add_levels_R=[1.0, 2.0, 3.0],
        add_notional_fracs=[0.6, 0.4, 0.3],

        # Exit
        trail_k_atr=3.0,
        enable_trail_after_R=1.0,
        time_stop_bars=12,
        need_reach_R_in_time=0.8,

        # Global risk
        daily_loss_limit=150.0,
        max_consecutive_losses=3,
        cooldown_bars=60,
    )

    print("\nStrategy Configuration:")
    print(f"  Risk per campaign:    {config.risk_cash_per_campaign:.0f}U (fixed max loss)")
    print(f"  Breakout lookback:    {config.breakout_lookback} bars")
    print(f"  Volume required:      {config.vol_ratio_entry:.1f}x average")
    print(f"  Stop (ATR mult):      {config.k_stop_atr:.1f}")
    print(f"  Stop range:           {config.stop_pct_min*100:.1f}% - {config.stop_pct_max*100:.1f}%")
    print(f"  Max adds:             {config.max_adds}")
    print(f"  Add levels (R):       {config.add_levels_R}")
    print(f"  Trailing (ATR mult):  {config.trail_k_atr:.1f}")
    print(f"  Time stop:            {config.time_stop_bars} bars")
    print(f"  Daily loss limit:     {config.daily_loss_limit:.0f}U")

    # Generate data
    timestamps, opens, highs, lows, closes, volumes = generate_trending_data(
        days=30, interval_minutes=1
    )

    # Create backtester
    backtester = FOMOBacktester(config)

    # Run backtest
    result = backtester.run_backtest(
        timestamps=timestamps,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        symbol="BTC-USDC"
    )

    # Print results
    backtester.print_result(result)

    return result


if __name__ == "__main__":
    run_fomo_backtest_demo()
