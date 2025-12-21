"""
仓位管理模块
负责管理持仓、风险控制、止损止盈
"""
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float  # 持仓数量
    entry_price: float  # 开仓价格
    current_price: float  # 当前价格
    unrealized_pnl: float  # 未实现盈亏
    unrealized_pnl_pct: float  # 未实现盈亏百分比
    margin: float  # 保证金
    leverage: float  # 杠杆倍数
    liquidation_price: float  # 爆仓价格
    stop_loss_price: Optional[float] = None  # 止损价格
    take_profit_price: Optional[float] = None  # 止盈价格
    trailing_stop_price: Optional[float] = None  # 追踪止损价格
    highest_price: Optional[float] = None  # 最高价格（用于追踪止损）
    lowest_price: Optional[float] = None  # 最低价格（用于追踪止损）
    created_at: int = 0  # 创建时间戳

    def update_price(self, current_price: float):
        """更新当前价格和盈亏"""
        self.current_price = current_price

        # 更新未实现盈亏
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price

        # 更新最高/最低价格（用于追踪止损）
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
        if self.lowest_price is None or current_price < self.lowest_price:
            self.lowest_price = current_price

    def should_stop_loss(self) -> bool:
        """检查是否触发止损"""
        if not self.stop_loss_price:
            return False

        if self.side == "LONG":
            return self.current_price <= self.stop_loss_price
        else:  # SHORT
            return self.current_price >= self.stop_loss_price

    def should_take_profit(self) -> bool:
        """检查是否触发止盈"""
        if not self.take_profit_price:
            return False

        if self.side == "LONG":
            return self.current_price >= self.take_profit_price
        else:  # SHORT
            return self.current_price <= self.take_profit_price

    def should_trailing_stop(self) -> bool:
        """检查是否触发追踪止损"""
        if not self.trailing_stop_price:
            return False

        if self.side == "LONG":
            return self.current_price <= self.trailing_stop_price
        else:  # SHORT
            return self.current_price >= self.trailing_stop_price

    def update_trailing_stop(self, trailing_pct: float):
        """更新追踪止损价格"""
        if self.side == "LONG" and self.highest_price:
            new_trailing_stop = self.highest_price * (1 - trailing_pct)
            if not self.trailing_stop_price or new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
                logger.info(f"{self.symbol} 多头追踪止损更新: {self.trailing_stop_price:.2f}")

        elif self.side == "SHORT" and self.lowest_price:
            new_trailing_stop = self.lowest_price * (1 + trailing_pct)
            if not self.trailing_stop_price or new_trailing_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
                logger.info(f"{self.symbol} 空头追踪止损更新: {self.trailing_stop_price:.2f}")

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class PositionManager:
    """仓位管理器"""

    def __init__(self, apex_exchange, strategy_config, risk_config):
        """
        初始化仓位管理器

        Args:
            apex_exchange: Apex交易所实例
            strategy_config: 策略配置
            risk_config: 风险配置
        """
        self.apex = apex_exchange
        self.strategy = strategy_config
        self.risk = risk_config
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.daily_stats = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "trades": 0,
            "pnl": 0.0,
            "starting_balance": 0.0,
        }

    def sync_positions(self):
        """从交易所同步持仓信息"""
        try:
            account_data = self.apex.account.get_account_data()

            if not account_data.get("success"):
                logger.error("获取账户数据失败")
                return

            data = account_data.get("data", {})
            positions_data = data.get("positions", [])

            # 更新或创建持仓
            current_symbols = set()
            for pos_data in positions_data:
                symbol = pos_data.get("symbol")
                size = float(pos_data.get("size", 0))

                if size == 0:
                    continue

                current_symbols.add(symbol)

                side = "LONG" if float(pos_data.get("size", 0)) > 0 else "SHORT"
                entry_price = float(pos_data.get("entryPrice", 0))
                current_price = float(pos_data.get("markPrice", 0))
                unrealized_pnl = float(pos_data.get("unrealizedPnl", 0))
                margin = float(pos_data.get("margin", 0))
                leverage = float(pos_data.get("leverage", 1))
                liquidation_price = float(pos_data.get("liquidationPrice", 0))

                if symbol in self.positions:
                    # 更新现有持仓
                    position = self.positions[symbol]
                    position.update_price(current_price)
                    position.size = abs(size)
                    position.unrealized_pnl = unrealized_pnl
                    position.margin = margin
                    position.leverage = leverage
                    position.liquidation_price = liquidation_price
                else:
                    # 创建新持仓
                    position = Position(
                        symbol=symbol,
                        side=side,
                        size=abs(size),
                        entry_price=entry_price,
                        current_price=current_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl / (margin * leverage) if margin > 0 else 0,
                        margin=margin,
                        leverage=leverage,
                        liquidation_price=liquidation_price,
                        created_at=int(time.time()),
                    )
                    self._set_stop_loss_take_profit(position)
                    self.positions[symbol] = position
                    logger.info(f"检测到新持仓: {symbol} {side} {size} @ {entry_price}")

            # 移除已平仓的持仓
            closed_symbols = set(self.positions.keys()) - current_symbols
            for symbol in closed_symbols:
                logger.info(f"持仓已关闭: {symbol}")
                del self.positions[symbol]

        except Exception as e:
            logger.error(f"同步持仓失败: {e}")

    def _set_stop_loss_take_profit(self, position: Position):
        """设置止损止盈价格"""
        if position.side == "LONG":
            position.stop_loss_price = position.entry_price * (1 - self.risk.stop_loss_pct)
            position.take_profit_price = position.entry_price * (1 + self.risk.take_profit_pct)
        else:  # SHORT
            position.stop_loss_price = position.entry_price * (1 + self.risk.stop_loss_pct)
            position.take_profit_price = position.entry_price * (1 - self.risk.take_profit_pct)

        logger.info(
            f"{position.symbol} 设置止损: {position.stop_loss_price:.2f}, "
            f"止盈: {position.take_profit_price:.2f}"
        )

    def check_risk_limits(self) -> Dict[str, List[str]]:
        """
        检查所有风险限制

        Returns:
            风险警告字典 {symbol: [warnings]}
        """
        warnings = {}

        # 检查每日亏损限制
        if self._check_daily_loss_limit():
            warnings["SYSTEM"] = ["达到每日最大亏损限制，停止交易"]

        # 检查每日交易次数限制
        if self._check_daily_trades_limit():
            if "SYSTEM" not in warnings:
                warnings["SYSTEM"] = []
            warnings["SYSTEM"].append("达到每日最大交易次数，停止交易")

        # 检查每个持仓
        for symbol, position in self.positions.items():
            position_warnings = []

            # 检查止损
            if position.should_stop_loss():
                position_warnings.append(f"触发止损 @ {position.stop_loss_price:.2f}")

            # 检查止盈
            if position.should_take_profit():
                position_warnings.append(f"触发止盈 @ {position.take_profit_price:.2f}")

            # 检查追踪止损
            if position.should_trailing_stop():
                position_warnings.append(f"触发追踪止损 @ {position.trailing_stop_price:.2f}")

            # 检查爆仓风险
            margin_ratio = self._calculate_margin_ratio(position)
            if margin_ratio and margin_ratio < self.risk.emergency_close_margin_ratio:
                position_warnings.append(f"紧急: 保证金率过低 ({margin_ratio:.2%}), 需要立即平仓!")
            elif margin_ratio and margin_ratio < self.risk.min_margin_ratio:
                position_warnings.append(f"警告: 保证金率偏低 ({margin_ratio:.2%})")

            if position_warnings:
                warnings[symbol] = position_warnings

        return warnings

    def _calculate_margin_ratio(self, position: Position) -> Optional[float]:
        """计算保证金率"""
        if position.margin <= 0:
            return None

        # 保证金率 = (保证金 + 未实现盈亏) / 持仓价值
        position_value = position.current_price * position.size
        if position_value <= 0:
            return None

        margin_ratio = (position.margin + position.unrealized_pnl) / position_value
        return margin_ratio

    def _check_daily_loss_limit(self) -> bool:
        """检查是否达到每日最大亏损"""
        self._update_daily_stats()

        if self.daily_stats["starting_balance"] <= 0:
            return False

        daily_loss_pct = abs(self.daily_stats["pnl"]) / self.daily_stats["starting_balance"]

        if self.daily_stats["pnl"] < 0 and daily_loss_pct >= self.risk.max_daily_loss_pct:
            logger.warning(
                f"达到每日最大亏损限制: {daily_loss_pct:.2%} "
                f"(限制: {self.risk.max_daily_loss_pct:.2%})"
            )
            return True

        return False

    def _check_daily_trades_limit(self) -> bool:
        """检查是否达到每日最大交易次数"""
        self._update_daily_stats()

        if self.daily_stats["trades"] >= self.risk.max_daily_trades:
            logger.warning(
                f"达到每日最大交易次数: {self.daily_stats['trades']} "
                f"(限制: {self.risk.max_daily_trades})"
            )
            return True

        return False

    def _update_daily_stats(self):
        """更新每日统计"""
        today = datetime.now().strftime("%Y-%m-%d")

        # 如果是新的一天，重置统计
        if self.daily_stats["date"] != today:
            logger.info(f"新的一天开始: {today}")
            self.daily_stats = {
                "date": today,
                "trades": 0,
                "pnl": 0.0,
                "starting_balance": self._get_account_balance(),
            }

    def _get_account_balance(self) -> float:
        """获取账户余额"""
        try:
            account_data = self.apex.account.get_account_data()
            if account_data.get("success"):
                data = account_data.get("data", {})
                return float(data.get("totalEquity", 0))
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
        return 0.0

    def update_trailing_stops(self):
        """更新所有持仓的追踪止损"""
        for position in self.positions.values():
            position.update_trailing_stop(self.risk.trailing_stop_pct)

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定交易对的持仓"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """检查是否持有某个交易对的仓位"""
        return symbol in self.positions

    def get_all_positions(self) -> List[Position]:
        """获取所有持仓"""
        return list(self.positions.values())

    def record_trade(self, pnl: float = 0.0):
        """记录交易"""
        self._update_daily_stats()
        self.daily_stats["trades"] += 1
        self.daily_stats["pnl"] += pnl
        logger.info(f"记录交易 - 今日交易: {self.daily_stats['trades']}, 今日盈亏: {self.daily_stats['pnl']:.2f}")

    def can_open_new_position(self, symbol: str) -> Tuple[bool, str]:
        """
        检查是否可以开新仓

        Returns:
            (是否可以, 原因)
        """
        # 检查每日限制
        if self._check_daily_loss_limit():
            return False, "达到每日最大亏损限制"

        if self._check_daily_trades_limit():
            return False, "达到每日最大交易次数"

        # 检查是否已有持仓
        if self.has_position(symbol):
            return False, f"{symbol} 已有持仓"

        # 检查账户余额
        balance = self._get_account_balance()
        if balance <= 0:
            return False, "账户余额不足"

        return True, "可以开仓"

    def save_state(self, filepath: str):
        """保存状态到文件"""
        try:
            state = {
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "daily_stats": self.daily_stats,
                "timestamp": int(time.time()),
            }
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"状态已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")

    def load_state(self, filepath: str):
        """从文件加载状态"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # 恢复持仓（但会在下次sync时重新同步）
            self.daily_stats = state.get("daily_stats", self.daily_stats)
            logger.info(f"状态已从 {filepath} 加载")
        except FileNotFoundError:
            logger.info(f"状态文件 {filepath} 不存在，使用初始状态")
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
