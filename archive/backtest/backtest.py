"""
回测系统
使用历史数据测试交易策略
"""
import os
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

# 技术指标计算（复制自market_analyzer.py，避免导入问题）
class TechnicalIndicators:
    """技术指标计算"""

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return float(np.mean(prices[-period:]))

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        prices_array = np.array(prices)
        ema = prices_array[0]
        multiplier = 2 / (period + 1)
        for price in prices_array[1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        prices_array = np.array(prices)
        deltas = np.diff(prices_array)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> Optional[float]:
        if len(prices) < period:
            return None
        returns = np.diff(prices[-period:]) / prices[-period:-1]
        return float(np.std(returns))


@dataclass
class BacktestConfig:
    """回测配置"""
    # 初始资金
    initial_capital: float = 10000.0

    # 策略参数
    ma_short_period: int = 10
    ma_long_period: int = 30
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_ma_period: int = 20
    volume_surge_ratio: float = 1.5
    min_signal_strength: float = 0.6

    # 风险参数
    max_position_pct: float = 0.10  # 最大仓位占资金比例
    stop_loss_pct: float = 0.02    # 止损比例
    take_profit_pct: float = 0.05  # 止盈比例
    trailing_stop_pct: float = 0.03  # 追踪止损比例

    # 交易成本
    commission_rate: float = 0.0006  # 手续费率 0.06%
    slippage: float = 0.0005  # 滑点 0.05%


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本统计
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # 盈亏统计
    total_profit: float
    total_loss: float
    profit_factor: float
    avg_profit: float
    avg_loss: float
    avg_trade_pnl: float

    # 风险统计
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # 时间统计
    start_date: str
    end_date: str
    trading_days: int

    # 交易明细
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class Backtester:
    """回测引擎"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.indicators = TechnicalIndicators()

    def generate_signal(
        self,
        closes: List[float],
        volumes: List[float],
        current_idx: int
    ) -> Tuple[str, float, str]:
        """
        生成交易信号

        Returns:
            (signal_type, strength, reason)
        """
        if current_idx < self.config.ma_long_period + 10:
            return "HOLD", 0, "Insufficient data"

        # 获取历史数据
        price_history = closes[:current_idx + 1]
        volume_history = volumes[:current_idx + 1]
        current_price = closes[current_idx]

        # 计算技术指标
        ma_short = self.indicators.calculate_sma(price_history, self.config.ma_short_period)
        ma_long = self.indicators.calculate_sma(price_history, self.config.ma_long_period)
        rsi = self.indicators.calculate_rsi(price_history, self.config.rsi_period)
        volume_ma = np.mean(volume_history[-self.config.volume_ma_period:])
        current_volume = volumes[current_idx]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0

        if not all([ma_short, ma_long, rsi]):
            return "HOLD", 0, "Indicator calc failed"

        # Signal scoring
        buy_score = 0
        sell_score = 0
        reasons = []

        # Trend signal - Moving Average
        if ma_short > ma_long:
            buy_score += 0.3
            reasons.append(f"MA Cross Up")
        elif ma_short < ma_long:
            sell_score += 0.3
            reasons.append(f"MA Cross Down")

        # Price position
        if current_price > ma_short:
            buy_score += 0.1
        elif current_price < ma_short:
            sell_score += 0.1

        # RSI signal
        if rsi < self.config.rsi_oversold:
            buy_score += 0.3
            reasons.append(f"RSI Oversold({rsi:.1f})")
        elif rsi > self.config.rsi_overbought:
            sell_score += 0.3
            reasons.append(f"RSI Overbought({rsi:.1f})")

        # Volume confirmation
        if volume_ratio > self.config.volume_surge_ratio:
            if buy_score > sell_score:
                buy_score += 0.2
                reasons.append(f"Volume({volume_ratio:.1f}x)")
            elif sell_score > buy_score:
                sell_score += 0.2
                reasons.append(f"Volume({volume_ratio:.1f}x)")

        # Mild signal
        if 40 < rsi < 60:
            if ma_short > ma_long and current_price > ma_short:
                buy_score += 0.1
            elif ma_short < ma_long and current_price < ma_short:
                sell_score += 0.1

        # Determine final signal
        if buy_score > sell_score and buy_score >= self.config.min_signal_strength:
            return "BUY", buy_score, "; ".join(reasons) if reasons else "Multi-indicator bullish"
        elif sell_score > buy_score and sell_score >= self.config.min_signal_strength:
            return "SELL", sell_score, "; ".join(reasons) if reasons else "Multi-indicator bearish"

        return "HOLD", 0, "No clear signal"

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
        """
        运行回测

        Args:
            timestamps: 时间戳列表
            opens: 开盘价列表
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            volumes: 成交量列表
            symbol: 交易对

        Returns:
            回测结果
        """
        print("=" * 60)
        print("Starting Backtest")
        print("=" * 60)
        print(f"Symbol: {symbol}")
        print(f"Data points: {len(closes)}")
        print(f"Period: {timestamps[0]} ~ {timestamps[-1]}")
        print(f"Initial Capital: {self.config.initial_capital:.2f} USDC")
        print("=" * 60)

        # Initialize
        capital = self.config.initial_capital
        position: Optional[Position] = None
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []
        peak_equity = capital

        # Stats
        max_drawdown = 0
        max_drawdown_pct = 0

        # Iterate through each candle
        for i in range(self.config.ma_long_period + 10, len(closes)):
            current_time = timestamps[i]
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Calculate current equity
            if position:
                if position.side == "LONG":
                    unrealized_pnl = (current_price - position.entry_price) * position.size
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.size
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

            # Check position for stop loss / take profit
            if position:
                should_close = False
                exit_reason = ""
                exit_price = current_price

                # Update trailing stop
                if position.side == "LONG":
                    if position.highest_price is None or current_high > position.highest_price:
                        position.highest_price = current_high
                        position.trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct)

                    # Check stop loss
                    if current_low <= position.stop_loss:
                        should_close = True
                        exit_reason = "Stop Loss"
                        exit_price = position.stop_loss
                    # Check trailing stop
                    elif position.trailing_stop and current_low <= position.trailing_stop:
                        should_close = True
                        exit_reason = "Trailing Stop"
                        exit_price = position.trailing_stop
                    # Check take profit
                    elif current_high >= position.take_profit:
                        should_close = True
                        exit_reason = "Take Profit"
                        exit_price = position.take_profit

                else:  # SHORT
                    if position.lowest_price is None or current_low < position.lowest_price:
                        position.lowest_price = current_low
                        position.trailing_stop = position.lowest_price * (1 + self.config.trailing_stop_pct)

                    # Check stop loss
                    if current_high >= position.stop_loss:
                        should_close = True
                        exit_reason = "Stop Loss"
                        exit_price = position.stop_loss
                    # Check trailing stop
                    elif position.trailing_stop and current_high >= position.trailing_stop:
                        should_close = True
                        exit_reason = "Trailing Stop"
                        exit_price = position.trailing_stop
                    # Check take profit
                    elif current_low <= position.take_profit:
                        should_close = True
                        exit_reason = "Take Profit"
                        exit_price = position.take_profit

                # Check reverse signal
                if not should_close:
                    signal, strength, reason = self.generate_signal(closes, volumes, i)
                    if position.side == "LONG" and signal == "SELL" and strength >= self.config.min_signal_strength:
                        should_close = True
                        exit_reason = f"Reverse: {reason}"
                    elif position.side == "SHORT" and signal == "BUY" and strength >= self.config.min_signal_strength:
                        should_close = True
                        exit_reason = f"Reverse: {reason}"

                # Execute close
                if should_close:
                    # Apply slippage
                    if position.side == "LONG":
                        exit_price = exit_price * (1 - self.config.slippage)
                        pnl = (exit_price - position.entry_price) * position.size
                    else:
                        exit_price = exit_price * (1 + self.config.slippage)
                        pnl = (position.entry_price - exit_price) * position.size

                    # Deduct commission
                    commission = exit_price * position.size * self.config.commission_rate
                    pnl -= commission

                    pnl_pct = pnl / (position.entry_price * position.size)

                    # Record trade
                    trade = Trade(
                        symbol=symbol,
                        side=position.side,
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        size=position.size,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)

                    # Update capital
                    capital += pnl
                    position = None

                    print(f"[{current_time}] CLOSE {trade.side} @ {exit_price:.2f} | "
                          f"PnL: {pnl:+.2f} ({pnl_pct:+.2%}) | {exit_reason}")

            # Check for new position if none exists
            if position is None:
                signal, strength, reason = self.generate_signal(closes, volumes, i)

                if signal in ["BUY", "SELL"]:
                    # Calculate position size
                    position_value = capital * self.config.max_position_pct
                    entry_price = current_price * (1 + self.config.slippage if signal == "BUY" else 1 - self.config.slippage)
                    size = position_value / entry_price

                    # Deduct entry commission
                    commission = position_value * self.config.commission_rate
                    capital -= commission

                    # Set stop loss / take profit
                    if signal == "BUY":
                        stop_loss = entry_price * (1 - self.config.stop_loss_pct)
                        take_profit = entry_price * (1 + self.config.take_profit_pct)
                        side = "LONG"
                    else:
                        stop_loss = entry_price * (1 + self.config.stop_loss_pct)
                        take_profit = entry_price * (1 - self.config.take_profit_pct)
                        side = "SHORT"

                    position = Position(
                        symbol=symbol,
                        side=side,
                        size=size,
                        entry_price=entry_price,
                        entry_time=current_time,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )

                    print(f"[{current_time}] OPEN {side} @ {entry_price:.2f} | "
                          f"SL: {stop_loss:.2f} TP: {take_profit:.2f} | {reason}")

        # Close remaining position at end
        if position:
            final_price = closes[-1]
            if position.side == "LONG":
                pnl = (final_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - final_price) * position.size

            commission = final_price * position.size * self.config.commission_rate
            pnl -= commission
            capital += pnl

            trade = Trade(
                symbol=symbol,
                side=position.side,
                entry_time=position.entry_time,
                exit_time=timestamps[-1],
                entry_price=position.entry_price,
                exit_price=final_price,
                size=position.size,
                pnl=pnl,
                pnl_pct=pnl / (position.entry_price * position.size),
                exit_reason="End of Backtest"
            )
            trades.append(trade)
            print(f"[{timestamps[-1]}] End of Backtest | PnL: {pnl:+.2f}")

        # Calculate statistics
        total_return = capital - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        avg_profit = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        avg_trade_pnl = total_return / len(trades) if trades else 0

        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Calculate Sharpe Ratio (simplified)
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                ret = (equity_curve[i][1] - equity_curve[i-1][1]) / equity_curve[i-1][1]
                returns.append(ret)
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

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
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            start_date=str(timestamps[0]),
            end_date=str(timestamps[-1]),
            trading_days=len(set(t.date() for t in timestamps)),
            trades=trades,
            equity_curve=equity_curve
        )

        return result

    def print_result(self, result: BacktestResult):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULT")
        print("=" * 60)

        print(f"\n[Time Range]")
        print(f"  Start: {result.start_date}")
        print(f"  End:   {result.end_date}")
        print(f"  Days:  {result.trading_days}")

        print(f"\n[Capital]")
        print(f"  Initial: {result.initial_capital:,.2f} USDC")
        print(f"  Final:   {result.final_capital:,.2f} USDC")
        print(f"  Return:  {result.total_return:+,.2f} USDC ({result.total_return_pct:+.2%})")

        print(f"\n[Trade Stats]")
        print(f"  Total Trades:   {result.total_trades}")
        print(f"  Winning Trades: {result.winning_trades}")
        print(f"  Losing Trades:  {result.losing_trades}")
        print(f"  Win Rate:       {result.win_rate:.2%}")

        print(f"\n[PnL Analysis]")
        print(f"  Total Profit:   {result.total_profit:,.2f} USDC")
        print(f"  Total Loss:     {result.total_loss:,.2f} USDC")
        print(f"  Profit Factor:  {result.profit_factor:.2f}")
        print(f"  Avg Profit:     {result.avg_profit:,.2f} USDC")
        print(f"  Avg Loss:       {result.avg_loss:,.2f} USDC")
        print(f"  Avg Trade PnL:  {result.avg_trade_pnl:+,.2f} USDC")

        print(f"\n[Risk Metrics]")
        print(f"  Max Drawdown:   {result.max_drawdown:,.2f} USDC ({result.max_drawdown_pct:.2%})")
        print(f"  Sharpe Ratio:   {result.sharpe_ratio:.2f}")

        # Evaluation
        print(f"\n[Strategy Evaluation]")
        if result.total_return_pct > 0:
            print(f"  [+] Strategy profitable: {result.total_return_pct:.2%}")
        else:
            print(f"  [-] Strategy loss: {result.total_return_pct:.2%}")

        if result.win_rate >= 0.5:
            print(f"  [+] Good win rate: {result.win_rate:.2%}")
        else:
            print(f"  [!] Low win rate: {result.win_rate:.2%}")

        if result.profit_factor >= 1.5:
            print(f"  [+] Excellent profit factor: {result.profit_factor:.2f}")
        elif result.profit_factor >= 1.0:
            print(f"  [=] Average profit factor: {result.profit_factor:.2f}")
        else:
            print(f"  [-] Poor profit factor: {result.profit_factor:.2f}")

        if result.max_drawdown_pct <= 0.10:
            print(f"  [+] Good drawdown control: {result.max_drawdown_pct:.2%}")
        elif result.max_drawdown_pct <= 0.20:
            print(f"  [!] Moderate drawdown: {result.max_drawdown_pct:.2%}")
        else:
            print(f"  [-] Large drawdown: {result.max_drawdown_pct:.2%}")

        print("\n" + "=" * 60)

        # Show recent trades
        if result.trades:
            print("\nRecent 10 Trades:")
            print("-" * 90)
            for trade in result.trades[-10:]:
                print(f"  {trade.entry_time.strftime('%m-%d %H:%M')} -> {trade.exit_time.strftime('%m-%d %H:%M')} | "
                      f"{trade.side:5} | {trade.entry_price:>10.2f} -> {trade.exit_price:<10.2f} | "
                      f"PnL: {trade.pnl:+8.2f} ({trade.pnl_pct:+6.2%}) | {trade.exit_reason}")


def generate_sample_data(days: int = 30, interval_minutes: int = 15) -> Tuple[List, List, List, List, List, List]:
    """
    Generate sample historical data for testing

    Args:
        days: number of days
        interval_minutes: candle interval in minutes

    Returns:
        (timestamps, opens, highs, lows, closes, volumes)
    """
    print("Generating sample data...")

    np.random.seed(42)  # Fixed seed for reproducibility

    # Calculate data points
    points_per_day = 24 * 60 // interval_minutes
    total_points = days * points_per_day

    # Generate timestamps
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=interval_minutes * i) for i in range(total_points)]

    # Generate price data (random walk + trend + cycle)
    base_price = 95000  # BTC base price

    # Generate returns
    returns = np.random.normal(0.0001, 0.02, total_points)  # Slightly positive mean, 2% std

    # Add some trend
    trend = np.sin(np.linspace(0, 4 * np.pi, total_points)) * 0.001

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + returns + trend)

    # Generate close prices
    closes = base_price * cumulative_returns

    # Generate OHLC
    opens = []
    highs = []
    lows = []

    for i, close in enumerate(closes):
        if i == 0:
            open_price = base_price
        else:
            open_price = closes[i - 1] * (1 + np.random.normal(0, 0.001))

        # Extend high/low from open/close range
        range_pct = abs(np.random.normal(0, 0.01))
        high = max(open_price, close) * (1 + range_pct)
        low = min(open_price, close) * (1 - range_pct)

        opens.append(open_price)
        highs.append(high)
        lows.append(low)

    # Generate volume (random + price change correlated)
    base_volume = 1000
    price_changes = np.abs(np.diff(closes, prepend=closes[0]))
    volumes = base_volume * (1 + price_changes / np.mean(price_changes)) * np.random.uniform(0.5, 1.5, total_points)

    print(f"Generated {total_points} data points")
    print(f"Price range: {min(closes):.2f} - {max(closes):.2f}")

    return timestamps, opens, highs, lows, list(closes), list(volumes)


def run_backtest_demo():
    """Run backtest demo"""
    print("=" * 60)
    print("Apex Trading Bot - Strategy Backtest")
    print("=" * 60)

    # Create backtest config
    config = BacktestConfig(
        initial_capital=10000.0,
        ma_short_period=10,
        ma_long_period=30,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        min_signal_strength=0.6,
        max_position_pct=0.10,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        trailing_stop_pct=0.03,
        commission_rate=0.0006,
        slippage=0.0005,
    )

    print("\nBacktest Config:")
    print(f"  Initial Capital:  {config.initial_capital:,.2f} USDC")
    print(f"  Max Position:     {config.max_position_pct * 100:.0f}%")
    print(f"  Stop Loss:        {config.stop_loss_pct * 100:.1f}%")
    print(f"  Take Profit:      {config.take_profit_pct * 100:.1f}%")
    print(f"  Trailing Stop:    {config.trailing_stop_pct * 100:.1f}%")
    print(f"  Signal Threshold: {config.min_signal_strength}")

    # Generate sample data
    timestamps, opens, highs, lows, closes, volumes = generate_sample_data(days=60, interval_minutes=15)

    # Create backtester
    backtester = Backtester(config)

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
    run_backtest_demo()
