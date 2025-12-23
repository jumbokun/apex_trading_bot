"""
跨币种中性对冲策略 - Cross-Asset Neutral Hedging Strategy
==========================================================

目的：通过高频调仓刷交易量，同时保持整体市场中性

策略逻辑：
1. 跨币种对冲：BTC多头 + ETH空头 + SOL多头（总价值接近中性）
2. 利用BTC/ETH/SOL的高相关性对冲市场风险
3. 定期调仓使总Delta接近0
4. 赚取Funding Rate差价 + 产生交易量

调仓机制：
- 目标：多头总价值 ≈ 空头总价值
- 当偏差超过阈值时调仓
- 通过增减仓位来平衡

示例配置：
- BTC LONG: $3000
- ETH SHORT: $3000
- SOL LONG: $3000
总Delta = +3000 + (-3000) + 3000 = +3000 (偏多)

调整后：
- BTC LONG: $2000
- ETH SHORT: $4000
- SOL LONG: $2000
总Delta = +2000 + (-4000) + 2000 = 0 (中性)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import time
import math


class PositionSide(Enum):
    LONG = "LONG"    # 多头
    SHORT = "SHORT"  # 空头


@dataclass
class AssetConfig:
    """单个资产配置"""
    symbol: str
    side: PositionSide
    target_notional: float  # 目标名义价值 (USDT)
    weight: float = 1.0     # 权重（用于调仓时的优先级）


@dataclass
class HedgingConfig:
    """中性对冲策略配置"""

    # 资产配置 - 默认: BTC多/ETH空/SOL空
    # 目标总仓位: 150,000U (多头75,000U + 空头75,000U)
    # BTC多头75,000U vs ETH空头37,500U + SOL空头37,500U
    assets: List[AssetConfig] = field(default_factory=lambda: [
        AssetConfig("BTC-USDT", PositionSide.LONG, 75000.0, weight=1.0),
        AssetConfig("ETH-USDT", PositionSide.SHORT, 37500.0, weight=1.0),
        AssetConfig("SOL-USDT", PositionSide.SHORT, 37500.0, weight=1.0),
    ])

    # 杠杆
    leverage: int = 3

    # 调仓参数 - 1分钟一次缓慢加仓
    rebalance_interval_seconds: int = 60  # 检查间隔（1分钟）
    delta_threshold_pct: float = 0.03  # Delta偏差阈值（3%）
    min_rebalance_interval_seconds: int = 60  # 最小调仓间隔（1分钟）
    max_rebalances_per_hour: int = 60  # 每小时最大调仓次数

    # 单次调仓幅度 - 每次2000U
    scale_up_amount: float = 2000.0  # 每次加仓金额（多空各1000U）
    rebalance_step_pct: float = 0.05  # 每次调仓5%
    min_trade_notional: float = 100.0  # 最小交易金额

    # 风险控制
    max_single_position_notional: float = 80000.0  # 单仓位上限
    max_total_exposure: float = 160000.0  # 总敞口上限
    emergency_delta_pct: float = 0.10  # 紧急Delta阈值（10%触发强制调仓）

    # 模式
    dry_run: bool = True  # 模拟模式


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    quantity: float = 0.0
    avg_price: float = 0.0
    target_notional: float = 0.0  # 目标名义价值
    last_update_time: float = 0.0

    def get_notional(self, current_price: float) -> float:
        """获取当前名义价值"""
        return self.quantity * current_price

    def get_delta(self, current_price: float) -> float:
        """获取Delta值（多头正，空头负）"""
        notional = self.get_notional(current_price)
        return notional if self.side == PositionSide.LONG else -notional


@dataclass
class RebalanceAction:
    """调仓操作"""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    notional: float
    reason: str

    def __repr__(self):
        return f"{self.symbol} {self.side} {self.quantity:.6f} (${self.notional:.0f}) - {self.reason}"


class NeutralHedgingStrategy:
    """跨币种中性对冲策略"""

    def __init__(self, config: HedgingConfig = None):
        self.config = config or HedgingConfig()
        self.positions: Dict[str, Position] = {}
        self.prices: Dict[str, float] = {}

        self.start_time = time.time()
        self.total_volume = 0.0
        self.rebalance_count = 0
        self.last_rebalance_time = 0.0
        self.rebalance_count_this_hour = 0
        self.last_hour_reset = time.time()
        self.is_paused = False

        # 初始化目标持仓
        for asset in self.config.assets:
            self.positions[asset.symbol] = Position(
                symbol=asset.symbol,
                side=asset.side,
                quantity=0.0,
                avg_price=0.0,
                target_notional=asset.target_notional
            )

    def update_price(self, symbol: str, price: float):
        """更新价格"""
        self.prices[symbol] = price

    def update_position(self, symbol: str, quantity: float, avg_price: float):
        """更新持仓"""
        if symbol in self.positions:
            self.positions[symbol].quantity = quantity
            self.positions[symbol].avg_price = avg_price
            self.positions[symbol].last_update_time = time.time()

    def get_portfolio_delta(self) -> Tuple[float, float, float]:
        """
        计算组合Delta

        返回: (总多头价值, 总空头价值, 净Delta)
        """
        total_long = 0.0
        total_short = 0.0

        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            if price == 0:
                continue

            notional = pos.get_notional(price)
            if pos.side == PositionSide.LONG:
                total_long += notional
            else:
                total_short += notional

        net_delta = total_long - total_short
        return total_long, total_short, net_delta

    def get_delta_imbalance_pct(self) -> float:
        """
        获取Delta偏差百分比

        偏差 = |净Delta| / 总敞口
        """
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short

        if total_exposure == 0:
            return 0.0

        return abs(net_delta) / total_exposure

    def is_neutral(self) -> bool:
        """检查是否处于中性状态"""
        return self.get_delta_imbalance_pct() < self.config.delta_threshold_pct

    def should_rebalance(self) -> Tuple[bool, str]:
        """
        判断是否需要调仓

        返回: (是否需要调仓, 原因)
        """
        if self.is_paused:
            return False, "Strategy paused"

        now = time.time()

        # 重置每小时计数
        if now - self.last_hour_reset >= 3600:
            self.rebalance_count_this_hour = 0
            self.last_hour_reset = now

        # 检查最小调仓间隔
        time_since_last = now - self.last_rebalance_time
        if time_since_last < self.config.min_rebalance_interval_seconds:
            return False, f"Too soon (wait {self.config.min_rebalance_interval_seconds - time_since_last:.0f}s)"

        # 检查每小时调仓次数
        if self.rebalance_count_this_hour >= self.config.max_rebalances_per_hour:
            return False, f"Hourly limit reached ({self.config.max_rebalances_per_hour})"

        # 检查是否有仓位（首次建仓）
        total_qty = sum(pos.quantity for pos in self.positions.values())
        if total_qty == 0:
            return True, "Initialize positions"

        # 检查Delta偏差
        imbalance_pct = self.get_delta_imbalance_pct()

        # 紧急调仓
        if imbalance_pct >= self.config.emergency_delta_pct:
            return True, f"EMERGENCY: Imbalance {imbalance_pct:.1%} >= {self.config.emergency_delta_pct:.1%}"

        # 常规阈值调仓
        if imbalance_pct >= self.config.delta_threshold_pct:
            return True, f"Imbalance {imbalance_pct:.1%} >= {self.config.delta_threshold_pct:.1%}"

        # 检查是否需要向目标仓位加仓（当前仓位远低于目标）
        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            if price > 0:
                current_notional = pos.get_notional(price)
                target_notional = pos.target_notional
                if target_notional > 0:
                    fill_ratio = current_notional / target_notional
                    # 如果当前仓位不足目标的90%，需要加仓
                    if fill_ratio < 0.90:
                        return True, f"Scale up {symbol} ({fill_ratio:.0%} of target)"

        # 时间间隔调仓（维持交易量）
        if time_since_last >= self.config.rebalance_interval_seconds:
            return True, f"Time interval ({time_since_last/60:.1f}min)"

        return False, "No rebalance needed"

    def calculate_rebalance_actions(self) -> List[RebalanceAction]:
        """
        计算调仓操作

        策略：
        1. 计算当前Delta偏差方向（偏多还是偏空）
        2. 如果偏多：增加空头或减少多头
        3. 如果偏空：增加多头或减少空头
        4. 优先调整离目标偏差最大的仓位
        """
        actions = []

        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short

        if total_exposure == 0:
            # 首次建仓：按目标建立所有仓位
            for symbol, pos in self.positions.items():
                price = self.prices.get(symbol, 0)
                if price == 0:
                    continue

                target_qty = pos.target_notional / price
                if target_qty > 0:
                    side = "BUY" if pos.side == PositionSide.LONG else "SELL"
                    actions.append(RebalanceAction(
                        symbol=symbol,
                        side=side,
                        quantity=target_qty,
                        notional=pos.target_notional,
                        reason="Initial position"
                    ))
            return actions

        imbalance_pct = abs(net_delta) / total_exposure

        # 首先检查是否需要向目标仓位加仓（delta中性加仓）
        # 找出所有低于目标的仓位，成对加仓（一个多头+一个空头）保持中性
        scale_up_actions = self._calculate_scale_up_actions()
        if scale_up_actions:
            return scale_up_actions

        # 计算需要调整的金额
        # 目标：使 total_long ≈ total_short
        # 如果 net_delta > 0 (偏多)：需要减少多头或增加空头
        # 如果 net_delta < 0 (偏空)：需要增加多头或减少空头

        adjustment_needed = abs(net_delta) / 2  # 调整一半即可达到平衡

        # 限制单次调整幅度
        max_adjustment = total_exposure * self.config.rebalance_step_pct
        adjustment = min(adjustment_needed, max_adjustment)

        if adjustment < self.config.min_trade_notional:
            return actions  # 调整金额太小，不值得交易

        if net_delta > 0:
            # 偏多：需要减少多头敞口或增加空头敞口
            # 策略：找最大的多头减仓，或找最小的空头加仓

            # 优先减少多头
            long_positions = [(s, p) for s, p in self.positions.items()
                             if p.side == PositionSide.LONG and p.quantity > 0]
            if long_positions:
                # 选择持仓最大的多头减仓
                long_positions.sort(key=lambda x: x[1].get_notional(self.prices.get(x[0], 0)),
                                   reverse=True)
                symbol, pos = long_positions[0]
                price = self.prices.get(symbol, 0)
                if price > 0:
                    reduce_qty = adjustment / price
                    reduce_qty = min(reduce_qty, pos.quantity * 0.5)  # 最多减一半
                    if reduce_qty * price >= self.config.min_trade_notional:
                        actions.append(RebalanceAction(
                            symbol=symbol,
                            side="SELL",  # 平多头
                            quantity=reduce_qty,
                            notional=reduce_qty * price,
                            reason=f"Reduce LONG (delta {net_delta:.0f} -> {net_delta - reduce_qty*price:.0f})"
                        ))

            # 也可以增加空头
            short_positions = [(s, p) for s, p in self.positions.items()
                              if p.side == PositionSide.SHORT]
            if short_positions and len(actions) == 0:
                # 选择持仓最小的空头加仓
                short_positions.sort(key=lambda x: x[1].get_notional(self.prices.get(x[0], 0)))
                symbol, pos = short_positions[0]
                price = self.prices.get(symbol, 0)
                if price > 0:
                    add_qty = adjustment / price
                    if add_qty * price >= self.config.min_trade_notional:
                        actions.append(RebalanceAction(
                            symbol=symbol,
                            side="SELL",  # 加空头
                            quantity=add_qty,
                            notional=add_qty * price,
                            reason=f"Add SHORT (delta {net_delta:.0f} -> {net_delta - add_qty*price:.0f})"
                        ))

        else:
            # 偏空：需要增加多头敞口或减少空头敞口
            # 优先增加多头
            long_positions = [(s, p) for s, p in self.positions.items()
                             if p.side == PositionSide.LONG]
            if long_positions:
                # 选择持仓最小的多头加仓
                long_positions.sort(key=lambda x: x[1].get_notional(self.prices.get(x[0], 0)))
                symbol, pos = long_positions[0]
                price = self.prices.get(symbol, 0)
                if price > 0:
                    add_qty = adjustment / price
                    if add_qty * price >= self.config.min_trade_notional:
                        actions.append(RebalanceAction(
                            symbol=symbol,
                            side="BUY",  # 加多头
                            quantity=add_qty,
                            notional=add_qty * price,
                            reason=f"Add LONG (delta {net_delta:.0f} -> {net_delta + add_qty*price:.0f})"
                        ))

            # 也可以减少空头
            short_positions = [(s, p) for s, p in self.positions.items()
                              if p.side == PositionSide.SHORT and p.quantity > 0]
            if short_positions and len(actions) == 0:
                # 选择持仓最大的空头减仓
                short_positions.sort(key=lambda x: x[1].get_notional(self.prices.get(x[0], 0)),
                                    reverse=True)
                symbol, pos = short_positions[0]
                price = self.prices.get(symbol, 0)
                if price > 0:
                    reduce_qty = adjustment / price
                    reduce_qty = min(reduce_qty, pos.quantity * 0.5)  # 最多减一半
                    if reduce_qty * price >= self.config.min_trade_notional:
                        actions.append(RebalanceAction(
                            symbol=symbol,
                            side="BUY",  # 平空头
                            quantity=reduce_qty,
                            notional=reduce_qty * price,
                            reason=f"Reduce SHORT (delta {net_delta:.0f} -> {net_delta + reduce_qty*price:.0f})"
                        ))

        return actions

    def _calculate_scale_up_actions(self) -> List[RebalanceAction]:
        """
        计算向目标仓位加仓的操作（保持delta中性）

        策略：
        1. 找出所有低于目标的仓位
        2. 成对加仓：多头和空头同时加相等金额，保持delta中性
        3. 每次固定加2000U（多头1000U + 空头1000U），1分钟一次
        """
        actions = []

        # 分离多头和空头仓位
        long_needs_scale = []  # (symbol, pos, current_notional, gap_to_target, price)
        short_needs_scale = []

        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            if price <= 0:
                continue

            current_notional = pos.get_notional(price)
            target_notional = pos.target_notional

            if target_notional <= 0:
                continue

            gap = target_notional - current_notional

            # 只要还没达到目标就需要加仓
            if gap >= self.config.min_trade_notional:
                if pos.side == PositionSide.LONG:
                    long_needs_scale.append((symbol, pos, current_notional, gap, price))
                else:
                    short_needs_scale.append((symbol, pos, current_notional, gap, price))

        # 如果没有需要加仓的仓位，返回空
        if not long_needs_scale and not short_needs_scale:
            return actions

        # 成对加仓：同时加多头和空头，保持中性
        # 优先选择gap最大的仓位
        long_needs_scale.sort(key=lambda x: x[3], reverse=True)
        short_needs_scale.sort(key=lambda x: x[3], reverse=True)

        # 固定每次加仓金额：2000U（多头1000U + 空头1000U）
        # 从配置读取 scale_up_amount，默认2000U
        total_scale_amount = getattr(self.config, 'scale_up_amount', 2000.0)
        per_side_amount = total_scale_amount / 2  # 多空各一半 = 1000U

        # 取加仓金额的较小值，保证两边加相等的量
        if long_needs_scale and short_needs_scale:
            # 两边都有需要加仓的，成对加
            long_item = long_needs_scale[0]
            short_item = short_needs_scale[0]

            # 使用固定金额，但不能超过gap
            long_add = min(per_side_amount, long_item[3])
            short_add = min(per_side_amount, short_item[3])

            # 取两者较小值，保证delta中性
            add_amount = min(long_add, short_add)

            # 确保加仓金额不会太小
            add_amount = max(add_amount, self.config.min_trade_notional)

            if add_amount >= self.config.min_trade_notional:
                # 加多头
                long_qty = add_amount / long_item[4]
                actions.append(RebalanceAction(
                    symbol=long_item[0],
                    side="BUY",
                    quantity=long_qty,
                    notional=add_amount,
                    reason=f"Scale up LONG +${add_amount:.0f}"
                ))

                # 加空头 - 如果有多个空头仓位，按比例分配
                if len(short_needs_scale) >= 2:
                    # 两个空头各加一半
                    half_amount = add_amount / 2
                    for short_item in short_needs_scale[:2]:
                        actual_add = min(half_amount, short_item[3])
                        if actual_add >= self.config.min_trade_notional / 2:
                            short_qty = actual_add / short_item[4]
                            actions.append(RebalanceAction(
                                symbol=short_item[0],
                                side="SELL",
                                quantity=short_qty,
                                notional=actual_add,
                                reason=f"Scale up SHORT +${actual_add:.0f}"
                            ))
                else:
                    # 只有一个空头
                    short_qty = add_amount / short_item[4]
                    actions.append(RebalanceAction(
                        symbol=short_item[0],
                        side="SELL",
                        quantity=short_qty,
                        notional=add_amount,
                        reason=f"Scale up SHORT +${add_amount:.0f}"
                    ))

        elif long_needs_scale:
            # 只有多头需要加仓（但这会破坏delta中性，需要谨慎）
            # 只有在当前偏空时才加多头
            total_long, total_short, net_delta = self.get_portfolio_delta()
            if net_delta < 0:  # 偏空，可以加多头
                long_item = long_needs_scale[0]
                add_amount = min(per_side_amount, abs(net_delta), long_item[3])
                if add_amount >= self.config.min_trade_notional:
                    long_qty = add_amount / long_item[4]
                    actions.append(RebalanceAction(
                        symbol=long_item[0],
                        side="BUY",
                        quantity=long_qty,
                        notional=add_amount,
                        reason=f"Scale up LONG (delta correction) +${add_amount:.0f}"
                    ))

        elif short_needs_scale:
            # 只有空头需要加仓
            # 只有在当前偏多时才加空头
            total_long, total_short, net_delta = self.get_portfolio_delta()
            if net_delta > 0:  # 偏多，可以加空头
                short_item = short_needs_scale[0]
                add_amount = min(per_side_amount, net_delta, short_item[3])
                if add_amount >= self.config.min_trade_notional:
                    short_qty = add_amount / short_item[4]
                    actions.append(RebalanceAction(
                        symbol=short_item[0],
                        side="SELL",
                        quantity=short_qty,
                        notional=add_amount,
                        reason=f"Scale up SHORT (delta correction) +${add_amount:.0f}"
                    ))

        return actions

    def record_rebalance(self, actions: List[RebalanceAction]):
        """记录调仓"""
        self.last_rebalance_time = time.time()
        self.rebalance_count += 1
        self.rebalance_count_this_hour += 1

        for action in actions:
            self.total_volume += action.notional

    def get_portfolio_summary(self) -> dict:
        """获取组合摘要"""
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short
        imbalance_pct = self.get_delta_imbalance_pct()

        positions_detail = []
        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            positions_detail.append({
                "symbol": symbol,
                "side": pos.side.value,
                "quantity": pos.quantity,
                "price": price,
                "notional": pos.get_notional(price),
                "delta": pos.get_delta(price),
                "target_notional": pos.target_notional
            })

        return {
            "total_long": total_long,
            "total_short": total_short,
            "net_delta": net_delta,
            "total_exposure": total_exposure,
            "imbalance_pct": imbalance_pct,
            "is_neutral": self.is_neutral(),
            "total_volume": self.total_volume,
            "rebalance_count": self.rebalance_count,
            "positions": positions_detail
        }


# 策略元信息
STRATEGY_NAME = "Cross-Asset Neutral Hedging"
STRATEGY_VERSION = "2.0.0"
STRATEGY_DESCRIPTION = """
跨币种中性对冲策略 - 用于刷交易量

核心机制：
1. BTC多头 + ETH空头 + SOL多头（可配置）
2. 保持总多头价值 ≈ 总空头价值
3. 通过高频调仓产生交易量
4. 利用币种相关性对冲市场风险

优势：
- 不需要同一币种开双向仓（节省保证金）
- 可以根据Funding Rate选择多空方向
- 灵活调整各币种敞口

示例配置：
- BTC LONG: $2000 (Delta +2000)
- ETH SHORT: $4000 (Delta -4000)
- SOL LONG: $2000 (Delta +2000)
- 净Delta = 0 (完美中性)
"""


if __name__ == "__main__":
    # 测试策略逻辑
    config = HedgingConfig()
    strategy = NeutralHedgingStrategy(config)

    # 模拟价格
    strategy.update_price("BTC-USDT", 95000.0)
    strategy.update_price("ETH-USDT", 3300.0)
    strategy.update_price("SOL-USDT", 180.0)

    # 检查是否需要调仓（首次应该需要建仓）
    should, reason = strategy.should_rebalance()
    print(f"Should rebalance: {should}, Reason: {reason}")

    if should:
        actions = strategy.calculate_rebalance_actions()
        print("\nRebalance actions:")
        for action in actions:
            print(f"  {action}")

    # 模拟建仓后
    strategy.update_position("BTC-USDT", 0.021, 95000.0)  # 2000 USDT
    strategy.update_position("ETH-USDT", 1.21, 3300.0)    # 4000 USDT (空头)
    strategy.update_position("SOL-USDT", 11.1, 180.0)     # 2000 USDT

    print("\n" + "="*50)
    print("Portfolio Summary after positions:")
    summary = strategy.get_portfolio_summary()
    print(f"  Total Long: ${summary['total_long']:.0f}")
    print(f"  Total Short: ${summary['total_short']:.0f}")
    print(f"  Net Delta: ${summary['net_delta']:.0f}")
    print(f"  Imbalance: {summary['imbalance_pct']:.2%}")
    print(f"  Is Neutral: {summary['is_neutral']}")
