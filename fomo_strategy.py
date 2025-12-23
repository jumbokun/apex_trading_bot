"""
FOMO策略 - 正式版本 (支持双向交易)
基于优化结果的最佳参数配置

核心逻辑:
1. 向上突破入场: 价格突破60根K线高点 + 成交量2倍确认 -> 开多
2. 向下突破入场: 价格跌破60根K线低点 + 成交量2倍确认 -> 开空
3. ATR止损: 1.5倍ATR止损
4. 追踪止盈: 盈利达到1.5R后启用5倍ATR追踪止损
5. 无时间止损: 让趋势充分发展

风险控制:
- 每笔交易最大亏损50U
- 每日最大亏损150U
- 单一品种冷却期
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class FOMOStrategyConfig:
    """FOMO策略配置 - 参数优化后(14天回测+815U)"""

    # 资金管理
    initial_capital: float = 5000.0
    risk_per_trade: float = 50.0  # 每笔交易最大风险

    # 交易方向
    enable_long: bool = True  # 是否允许做多
    enable_short: bool = False  # 是否允许做空 (默认关闭，回测显示空头表现差)

    # 入场条件 - 使用15分钟K线
    breakout_lookback: int = 20  # 突破回看周期 (15min线20根=5小时)
    breakout_buffer_atr: float = 0.4  # 突破缓冲(ATR倍数)
    vol_ratio_entry: float = 2.0  # 入场成交量要求(均值倍数)
    vol_ma_period: int = 20  # 成交量均线周期

    # 止损 - 放宽止损距离
    atr_period: int = 14  # ATR周期
    k_stop_atr: float = 2.0  # 止损距离(ATR倍数) - 从1.5提高到2.0
    stop_pct_min: float = 0.02  # 最小止损百分比 - 从1%提高到2%
    stop_pct_max: float = 0.08  # 最大止损百分比 - 从6%提高到8%

    # 追踪止损 - 优化后参数
    trail_k_atr: float = 7.0  # 追踪止损距离(ATR倍数) - 从5.0优化到7.0
    enable_trail_after_R: float = 3.0  # 启用追踪的R倍数 - 从1.5优化到3.0

    # 杠杆
    leverage_min: int = 5
    leverage_max: int = 20

    # 风控
    daily_loss_limit: float = 150.0  # 每日亏损限额
    max_consecutive_losses: int = 5  # 最大连续亏损次数
    cooldown_minutes: int = 30  # 亏损后冷却时间(分钟)
    symbol_cooldown_minutes: int = 60  # 同品种冷却时间
    max_positions: int =  10  # 最大同时持仓数

    # 波动率调整仓位 - 降低仓位大小
    # 低波动率(止损<2%) -> 较大仓位, 高波动率(止损>4%) -> 较小仓位
    notional_min: float = 100.0  # 最小名义价值 (从500降到100)
    notional_max: float = 500.0  # 最大名义价值 (从2500降到500)

    # 执行
    commission_rate: float = 0.0006  # 手续费率
    slippage: float = 0.001  # 滑点


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    avg_price: float
    quantity: float
    stop_price: float
    R_value: float  # 1R = |entry - stop|
    highest_price: float  # 多头用
    lowest_price: float = 0.0  # 空头用
    trail_stop: Optional[float] = None
    trail_enabled: bool = False
    entry_time: datetime = None
    entry_atr: float = 0.0


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    action: str  # "OPEN_LONG", "OPEN_SHORT", "CLOSE", "NONE"
    price: float
    stop_price: float = 0.0
    quantity: float = 0.0
    reason: str = ""
    side: str = ""  # "BUY" or "SELL" for exchange API


class TechnicalAnalysis:
    """技术分析工具"""

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float],
                      closes: List[float], period: int = 14) -> Optional[float]:
        """计算ATR"""
        if len(highs) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        return np.mean(true_ranges[-period:])

    @staticmethod
    def get_highest(data: List[float], period: int) -> float:
        """获取最高值"""
        if len(data) < period:
            return max(data) if data else 0
        return max(data[-period:])

    @staticmethod
    def get_lowest(data: List[float], period: int) -> float:
        """获取最低值"""
        if len(data) < period:
            return min(data) if data else float('inf')
        return min(data[-period:])

    @staticmethod
    def get_volume_ratio(volumes: List[float], current_idx: int,
                         ma_period: int = 60) -> float:
        """计算成交量比率"""
        if current_idx < ma_period:
            return 0

        vol_ma = np.mean(volumes[current_idx - ma_period:current_idx])
        if vol_ma <= 0:
            return 0

        return volumes[current_idx] / vol_ma


class FOMOStrategy:
    """
    FOMO交易策略

    使用方法:
    1. 初始化策略
    2. 每根K线调用 update() 更新数据
    3. 调用 generate_signal() 获取交易信号
    4. 根据信号执行交易
    5. 交易执行后调用 on_trade_executed() 更新状态
    """

    def __init__(self, config: FOMOStrategyConfig = None):
        self.config = config or FOMOStrategyConfig()
        self.ta = TechnicalAnalysis()

        # 数据缓存
        self.timestamps: List[datetime] = []
        self.opens: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.volumes: List[float] = []

        # 状态
        self.position: Optional[Position] = None
        self.capital: float = self.config.initial_capital
        self.daily_pnl: float = 0.0
        self.last_trade_date: Optional[datetime] = None
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None
        self.symbol_cooldowns: Dict[str, datetime] = {}

        # 统计
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0

    def update(self, timestamp: datetime, open_: float, high: float,
               low: float, close: float, volume: float):
        """更新K线数据"""
        # 检查是否新的一天
        if self.last_trade_date and timestamp.date() != self.last_trade_date.date():
            self.daily_pnl = 0.0

        self.last_trade_date = timestamp

        self.timestamps.append(timestamp)
        self.opens.append(open_)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)

        # 保持数据窗口大小
        max_len = max(self.config.breakout_lookback, self.config.vol_ma_period,
                      self.config.atr_period) + 100
        if len(self.closes) > max_len:
            self.timestamps = self.timestamps[-max_len:]
            self.opens = self.opens[-max_len:]
            self.highs = self.highs[-max_len:]
            self.lows = self.lows[-max_len:]
            self.closes = self.closes[-max_len:]
            self.volumes = self.volumes[-max_len:]

    def generate_signal(self, symbol: str) -> Signal:
        """生成交易信号"""
        if len(self.closes) < self.config.breakout_lookback + 10:
            return Signal(symbol=symbol, action="NONE", price=0, reason="数据不足")

        current_price = self.closes[-1]
        current_high = self.highs[-1]
        current_low = self.lows[-1]
        current_time = self.timestamps[-1]

        # 计算ATR
        atr = self.ta.calculate_atr(
            self.highs, self.lows, self.closes, self.config.atr_period
        )
        if not atr:
            return Signal(symbol=symbol, action="NONE", price=0, reason="无法计算ATR")

        # ========== 持仓管理 ==========
        if self.position:
            return self._manage_position(symbol, current_price, current_high,
                                         current_low, atr)

        # ========== 新开仓检查 ==========
        return self._check_entry(symbol, current_price, current_high,
                                 current_time, atr)

    def _manage_position(self, symbol: str, price: float, high: float,
                         low: float, atr: float) -> Signal:
        """管理现有持仓 (支持多头和空头)"""
        pos = self.position

        if pos.side == "LONG":
            # ===== 多头管理 =====
            # 更新最高价
            if high > pos.highest_price:
                pos.highest_price = high

            # 计算当前盈亏R倍数
            profit_R = (price - pos.entry_price) / pos.R_value

            # 启用追踪止损
            if not pos.trail_enabled and profit_R >= self.config.enable_trail_after_R:
                pos.trail_enabled = True
                pos.trail_stop = pos.highest_price - self.config.trail_k_atr * atr

            # 更新追踪止损
            if pos.trail_enabled:
                new_trail = pos.highest_price - self.config.trail_k_atr * atr
                if pos.trail_stop is None or new_trail > pos.trail_stop:
                    pos.trail_stop = new_trail

            # 确定有效止损价
            effective_stop = pos.stop_price
            if pos.trail_stop and pos.trail_stop > effective_stop:
                effective_stop = pos.trail_stop

            # 检查是否触发止损 (多头: 价格跌破止损)
            if low <= effective_stop:
                reason = "追踪止损" if pos.trail_enabled else "止损"
                return Signal(
                    symbol=symbol,
                    action="CLOSE",
                    price=effective_stop,
                    reason=f"多头{reason} @ {effective_stop:.2f}",
                    side="SELL"  # 平多
                )

        else:  # SHORT
            # ===== 空头管理 =====
            # 更新最低价
            if low < pos.lowest_price or pos.lowest_price == 0:
                pos.lowest_price = low

            # 计算当前盈亏R倍数 (空头: 入场价 - 当前价)
            profit_R = (pos.entry_price - price) / pos.R_value

            # 启用追踪止损
            if not pos.trail_enabled and profit_R >= self.config.enable_trail_after_R:
                pos.trail_enabled = True
                pos.trail_stop = pos.lowest_price + self.config.trail_k_atr * atr

            # 更新追踪止损 (空头: 追踪止损跟随最低价向下移动)
            if pos.trail_enabled:
                new_trail = pos.lowest_price + self.config.trail_k_atr * atr
                if pos.trail_stop is None or new_trail < pos.trail_stop:
                    pos.trail_stop = new_trail

            # 确定有效止损价 (空头止损在上方)
            effective_stop = pos.stop_price
            if pos.trail_stop and pos.trail_stop < effective_stop:
                effective_stop = pos.trail_stop

            # 检查是否触发止损 (空头: 价格涨破止损)
            if high >= effective_stop:
                reason = "追踪止损" if pos.trail_enabled else "止损"
                return Signal(
                    symbol=symbol,
                    action="CLOSE",
                    price=effective_stop,
                    reason=f"空头{reason} @ {effective_stop:.2f}",
                    side="BUY"  # 平空
                )

        return Signal(symbol=symbol, action="NONE", price=0,
                      reason=f"{'多' if pos.side == 'LONG' else '空'}头持仓中")

    def _check_entry(self, symbol: str, price: float, high: float,
                     current_time: datetime, atr: float) -> Signal:
        """检查是否满足开仓条件 (支持做多和做空)"""

        # 风控检查
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return Signal(symbol=symbol, action="NONE", price=0, reason="达到日亏损限额")

        if self.cooldown_until and current_time < self.cooldown_until:
            return Signal(symbol=symbol, action="NONE", price=0, reason="冷却期中")

        if symbol in self.symbol_cooldowns:
            if current_time < self.symbol_cooldowns[symbol]:
                return Signal(symbol=symbol, action="NONE", price=0, reason="品种冷却期")

        # 数据检查
        lookback = self.config.breakout_lookback
        if len(self.highs) < lookback + 1:
            return Signal(symbol=symbol, action="NONE", price=0, reason="数据不足")

        # 计算突破水平
        highest = max(self.highs[-lookback-1:-1])  # 不包含当前K线
        lowest = min(self.lows[-lookback-1:-1])  # 不包含当前K线
        upper_breakout = highest + self.config.breakout_buffer_atr * atr
        lower_breakout = lowest - self.config.breakout_buffer_atr * atr

        # 成交量检查
        vol_ratio = self.ta.get_volume_ratio(
            self.volumes, len(self.volumes) - 1, self.config.vol_ma_period
        )

        # 判断突破方向 (根据配置决定是否启用)
        is_long_breakout = self.config.enable_long and price > upper_breakout
        is_short_breakout = self.config.enable_short and price < lower_breakout

        if not is_long_breakout and not is_short_breakout:
            return Signal(symbol=symbol, action="NONE", price=0, reason="未突破")

        # 成交量确认
        if vol_ratio < self.config.vol_ratio_entry:
            return Signal(symbol=symbol, action="NONE", price=0,
                         reason=f"成交量不足 ({vol_ratio:.1f}x < {self.config.vol_ratio_entry}x)")

        # 计算止损百分比
        stop_pct = self.config.k_stop_atr * atr / price
        stop_pct = max(self.config.stop_pct_min,
                       min(stop_pct, self.config.stop_pct_max))

        if stop_pct >= self.config.stop_pct_max:
            return Signal(symbol=symbol, action="NONE", price=0, reason="波动率过高")

        # 根据波动率调整仓位大小
        # 低波动率(1%) -> 最大仓位, 高波动率(6%) -> 最小仓位
        # 线性插值: notional = max - (stop_pct - min) / (max - min) * (notional_max - notional_min)
        vol_range = self.config.stop_pct_max - self.config.stop_pct_min
        vol_ratio_normalized = (stop_pct - self.config.stop_pct_min) / vol_range
        notional = self.config.notional_max - vol_ratio_normalized * (self.config.notional_max - self.config.notional_min)
        notional = max(self.config.notional_min, min(notional, self.config.notional_max))

        # 检查保证金
        leverage = self._calculate_leverage(stop_pct)
        required_margin = notional / leverage
        if required_margin > self.capital * 0.5:
            return Signal(symbol=symbol, action="NONE", price=0, reason="资金不足")

        if is_long_breakout:
            # ===== 做多 =====
            entry_price = price * (1 + self.config.slippage)
            stop_price = entry_price * (1 - stop_pct)
            quantity = notional / entry_price

            return Signal(
                symbol=symbol,
                action="OPEN_LONG",
                price=entry_price,
                stop_price=stop_price,
                quantity=quantity,
                reason=f"向上突破 Vol:{vol_ratio:.1f}x Stop:{stop_pct*100:.1f}%",
                side="BUY"
            )
        else:
            # ===== 做空 =====
            entry_price = price * (1 - self.config.slippage)  # 空头滑点向下
            stop_price = entry_price * (1 + stop_pct)  # 空头止损在上方
            quantity = notional / entry_price

            return Signal(
                symbol=symbol,
                action="OPEN_SHORT",
                price=entry_price,
                stop_price=stop_price,
                quantity=quantity,
                reason=f"向下突破 Vol:{vol_ratio:.1f}x Stop:{stop_pct*100:.1f}%",
                side="SELL"
            )

    def _calculate_leverage(self, stop_pct: float) -> int:
        """计算杠杆倍数"""
        # 确保爆仓距离 >= 4倍止损距离
        L_cap = int(1 / (4 * stop_pct))
        return max(self.config.leverage_min,
                   min(L_cap, self.config.leverage_max))

    def on_trade_executed(self, symbol: str, action: str, price: float,
                          quantity: float, timestamp: datetime):
        """交易执行后回调 (支持多头和空头)"""
        if action == "OPEN_LONG":
            atr = self.ta.calculate_atr(
                self.highs, self.lows, self.closes, self.config.atr_period
            ) or 0

            stop_pct = self.config.k_stop_atr * atr / price
            stop_pct = max(self.config.stop_pct_min,
                           min(stop_pct, self.config.stop_pct_max))
            stop_price = price * (1 - stop_pct)

            self.position = Position(
                symbol=symbol,
                side="LONG",
                entry_price=price,
                avg_price=price,
                quantity=quantity,
                stop_price=stop_price,
                R_value=price - stop_price,
                highest_price=price,
                lowest_price=price,
                entry_time=timestamp,
                entry_atr=atr
            )

            # 扣除手续费
            commission = price * quantity * self.config.commission_rate
            self.capital -= commission

        elif action == "OPEN_SHORT":
            atr = self.ta.calculate_atr(
                self.highs, self.lows, self.closes, self.config.atr_period
            ) or 0

            stop_pct = self.config.k_stop_atr * atr / price
            stop_pct = max(self.config.stop_pct_min,
                           min(stop_pct, self.config.stop_pct_max))
            stop_price = price * (1 + stop_pct)  # 空头止损在上方

            self.position = Position(
                symbol=symbol,
                side="SHORT",
                entry_price=price,
                avg_price=price,
                quantity=quantity,
                stop_price=stop_price,
                R_value=stop_price - price,  # 空头的R值
                highest_price=price,
                lowest_price=price,
                entry_time=timestamp,
                entry_atr=atr
            )

            # 扣除手续费
            commission = price * quantity * self.config.commission_rate
            self.capital -= commission

        elif action == "CLOSE" and self.position:
            # 计算盈亏 (区分多空)
            if self.position.side == "LONG":
                pnl = (price - self.position.avg_price) * self.position.quantity
            else:  # SHORT
                pnl = (self.position.avg_price - price) * self.position.quantity

            commission = price * self.position.quantity * self.config.commission_rate
            pnl -= commission

            self.capital += pnl
            self.daily_pnl += pnl
            self.total_pnl += pnl
            self.total_trades += 1

            if pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                # 设置冷却期
                if self.consecutive_losses >= self.config.max_consecutive_losses:
                    from datetime import timedelta
                    self.cooldown_until = timestamp + timedelta(
                        minutes=self.config.cooldown_minutes
                    )
                # 设置品种冷却
                from datetime import timedelta
                self.symbol_cooldowns[symbol] = timestamp + timedelta(
                    minutes=self.config.symbol_cooldown_minutes
                )

            self.position = None

    def get_status(self) -> Dict:
        """获取策略状态"""
        return {
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            "consecutive_losses": self.consecutive_losses,
            "has_position": self.position is not None,
            "position": {
                "symbol": self.position.symbol,
                "side": self.position.side,
                "entry_price": self.position.entry_price,
                "quantity": self.position.quantity,
                "stop_price": self.position.stop_price,
                "trail_stop": self.position.trail_stop,
                "trail_enabled": self.position.trail_enabled,
            } if self.position else None
        }


def test_strategy():
    """测试策略"""
    print("FOMO策略测试")
    print("=" * 50)

    config = FOMOStrategyConfig()
    strategy = FOMOStrategy(config)

    print(f"配置:")
    print(f"  突破周期: {config.breakout_lookback}")
    print(f"  成交量要求: {config.vol_ratio_entry}x")
    print(f"  止损ATR: {config.k_stop_atr}")
    print(f"  追踪ATR: {config.trail_k_atr}")
    print(f"  追踪启用: {config.enable_trail_after_R}R")

    print("\n策略已初始化，准备运行")
    print(f"初始资金: {strategy.capital}")


if __name__ == "__main__":
    test_strategy()
