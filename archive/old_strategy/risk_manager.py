"""
风险管理模块
负责止损止盈执行、爆仓保护、紧急平仓
"""
from typing import Dict, List, Optional
from loguru import logger
from .position_manager import Position, PositionManager


class RiskManager:
    """风险管理器"""

    def __init__(self, apex_exchange, position_manager: PositionManager, risk_config, dry_run: bool = True):
        """
        初始化风险管理器

        Args:
            apex_exchange: Apex交易所实例
            position_manager: 仓位管理器
            risk_config: 风险配置
            dry_run: 是否模拟运行
        """
        self.apex = apex_exchange
        self.position_manager = position_manager
        self.risk = risk_config
        self.dry_run = dry_run

    def monitor_positions(self) -> Dict[str, List[str]]:
        """
        监控所有持仓的风险

        Returns:
            执行的风险控制操作 {symbol: [actions]}
        """
        actions = {}

        # 同步持仓
        self.position_manager.sync_positions()

        # 更新追踪止损
        self.position_manager.update_trailing_stops()

        # 检查风险限制
        warnings = self.position_manager.check_risk_limits()

        # 处理系统级警告
        if "SYSTEM" in warnings:
            logger.warning(f"系统风险警告: {warnings['SYSTEM']}")
            # 如果触发系统级风险，可能需要平掉所有仓位
            if "达到每日最大亏损限制" in warnings["SYSTEM"]:
                logger.critical("达到每日最大亏损限制，准备平掉所有仓位")
                actions["SYSTEM"] = self._close_all_positions("达到每日亏损限制")

        # 检查每个持仓
        for symbol, position_warnings in warnings.items():
            if symbol == "SYSTEM":
                continue

            position = self.position_manager.get_position(symbol)
            if not position:
                continue

            position_actions = []

            # 检查爆仓保护
            if any("紧急" in w for w in position_warnings):
                logger.critical(f"{symbol} 触发紧急平仓保护")
                if self._close_position(position, "紧急平仓保护"):
                    position_actions.append("紧急平仓")

            # 检查止损
            elif position.should_stop_loss():
                logger.warning(f"{symbol} 触发止损 @ {position.stop_loss_price:.2f}")
                if self._close_position(position, f"止损 @ {position.stop_loss_price:.2f}"):
                    position_actions.append("止损平仓")

            # 检查追踪止损
            elif position.should_trailing_stop():
                logger.info(f"{symbol} 触发追踪止损 @ {position.trailing_stop_price:.2f}")
                if self._close_position(position, f"追踪止损 @ {position.trailing_stop_price:.2f}"):
                    position_actions.append("追踪止损平仓")

            # 检查止盈
            elif position.should_take_profit():
                logger.info(f"{symbol} 触发止盈 @ {position.take_profit_price:.2f}")
                if self._close_position(position, f"止盈 @ {position.take_profit_price:.2f}"):
                    position_actions.append("止盈平仓")

            if position_actions:
                actions[symbol] = position_actions

        return actions

    def _close_position(self, position: Position, reason: str) -> bool:
        """
        平仓

        Args:
            position: 持仓对象
            reason: 平仓原因

        Returns:
            是否成功
        """
        try:
            logger.info(f"准备平仓 {position.symbol}: {reason}")
            logger.info(
                f"持仓详情 - 方向: {position.side}, 数量: {position.size}, "
                f"开仓价: {position.entry_price:.2f}, 当前价: {position.current_price:.2f}, "
                f"盈亏: {position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2%})"
            )

            if self.dry_run:
                logger.info(f"[模拟模式] 平仓 {position.symbol} - {reason}")
                self.position_manager.record_trade(position.unrealized_pnl)
                return True

            # 实盘模式 - 市价平仓
            side = "SELL" if position.side == "LONG" else "BUY"

            order = self.apex.trade.create_order(
                symbol=position.symbol,
                side=side,
                type="MARKET",
                size=str(position.size),
                reduce_only=True,  # 只减仓
                client_order_id=f"close_{position.symbol}_{int(position.created_at)}",
            )

            if order.get("success"):
                logger.info(f"平仓成功: {position.symbol} - {reason}")
                logger.info(f"订单详情: {order}")
                self.position_manager.record_trade(position.unrealized_pnl)
                return True
            else:
                logger.error(f"平仓失败: {position.symbol} - {order}")
                return False

        except Exception as e:
            logger.error(f"平仓时发生错误 {position.symbol}: {e}")
            return False

    def _close_all_positions(self, reason: str) -> List[str]:
        """
        平掉所有持仓

        Args:
            reason: 平仓原因

        Returns:
            操作列表
        """
        actions = []
        positions = self.position_manager.get_all_positions()

        for position in positions:
            if self._close_position(position, reason):
                actions.append(f"平仓 {position.symbol}")

        return actions

    def check_position_size(self, symbol: str, size: float, price: float) -> bool:
        """
        检查仓位大小是否符合风险限制

        Args:
            symbol: 交易对
            size: 仓位大小
            price: 价格

        Returns:
            是否符合限制
        """
        # 获取账户余额
        balance = self.position_manager._get_account_balance()
        if balance <= 0:
            logger.warning("账户余额不足")
            return False

        # 计算仓位价值
        position_value = size * price

        # 检查仓位占比
        position_ratio = position_value / balance
        if position_ratio > self.risk.max_position_size:
            logger.warning(
                f"{symbol} 仓位过大: {position_ratio:.2%} > {self.risk.max_position_size:.2%}"
            )
            return False

        return True

    def calculate_position_size(self, symbol: str, price: float, signal_strength: float = 1.0) -> float:
        """
        根据账户余额和风险配置计算合适的仓位大小

        Args:
            symbol: 交易对
            price: 当前价格
            signal_strength: 信号强度 (0-1)

        Returns:
            仓位数量
        """
        # 获取账户余额
        balance = self.position_manager._get_account_balance()
        if balance <= 0:
            logger.warning("账户余额不足")
            return 0.0

        # 基础仓位金额
        base_position_value = balance * self.risk.max_position_size

        # 根据信号强度调整
        adjusted_position_value = base_position_value * signal_strength

        # 计算仓位数量
        size = adjusted_position_value / price

        # 获取交易对配置以检查最小下单量
        try:
            config_data = self.apex.public.get_all_config_data()
            if config_data.get("success"):
                symbols_config = config_data.get("data", {}).get("symbols", [])
                for symbol_config in symbols_config:
                    if symbol_config.get("symbol") == symbol:
                        min_size = float(symbol_config.get("minOrderSize", 0))
                        if size < min_size:
                            logger.warning(
                                f"{symbol} 计算仓位 {size} 小于最小下单量 {min_size}"
                            )
                            return 0.0
                        break
        except Exception as e:
            logger.warning(f"获取交易对配置失败: {e}")

        logger.info(
            f"{symbol} 计算仓位: 余额={balance:.2f}, "
            f"仓位价值={adjusted_position_value:.2f}, "
            f"数量={size:.6f}, 信号强度={signal_strength:.2f}"
        )

        return size

    def should_scale_in(self, position: Position) -> bool:
        """
        检查是否应该加仓

        Args:
            position: 当前持仓

        Returns:
            是否应该加仓
        """
        # 只在盈利时加仓
        if position.unrealized_pnl_pct < 0.01:  # 盈利至少1%
            return False

        # 检查仓位大小限制
        balance = self.position_manager._get_account_balance()
        if balance <= 0:
            return False

        current_position_value = position.size * position.current_price
        position_ratio = current_position_value / balance

        # 当前仓位小于最大仓位的80%才允许加仓
        if position_ratio >= self.risk.max_position_size * 0.8:
            logger.info(f"{position.symbol} 仓位已达上限，不加仓")
            return False

        return True

    def should_scale_out(self, position: Position) -> bool:
        """
        检查是否应该减仓

        Args:
            position: 当前持仓

        Returns:
            是否应该减仓
        """
        # 在亏损时减仓
        if position.unrealized_pnl_pct < -0.005:  # 亏损超过0.5%
            logger.info(f"{position.symbol} 亏损 {position.unrealized_pnl_pct:.2%}，建议减仓")
            return True

        # 在盈利较大时部分获利
        if position.unrealized_pnl_pct > 0.03:  # 盈利超过3%
            logger.info(f"{position.symbol} 盈利 {position.unrealized_pnl_pct:.2%}，建议部分获利")
            return True

        return False

    def get_risk_report(self) -> Dict:
        """
        生成风险报告

        Returns:
            风险报告字典
        """
        positions = self.position_manager.get_all_positions()
        balance = self.position_manager._get_account_balance()

        total_position_value = sum(p.size * p.current_price for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_margin = sum(p.margin for p in positions)

        report = {
            "账户余额": balance,
            "持仓数量": len(positions),
            "总持仓价值": total_position_value,
            "总未实现盈亏": total_unrealized_pnl,
            "总保证金": total_margin,
            "仓位占比": total_position_value / balance if balance > 0 else 0,
            "今日交易次数": self.position_manager.daily_stats["trades"],
            "今日盈亏": self.position_manager.daily_stats["pnl"],
            "持仓详情": [],
        }

        for position in positions:
            margin_ratio = self.position_manager._calculate_margin_ratio(position)
            report["持仓详情"].append({
                "交易对": position.symbol,
                "方向": position.side,
                "数量": position.size,
                "开仓价": position.entry_price,
                "当前价": position.current_price,
                "未实现盈亏": position.unrealized_pnl,
                "盈亏百分比": position.unrealized_pnl_pct,
                "保证金率": margin_ratio,
                "止损价": position.stop_loss_price,
                "止盈价": position.take_profit_price,
                "追踪止损价": position.trailing_stop_price,
            })

        return report
