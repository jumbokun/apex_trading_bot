"""
交易引擎
负责策略执行、订单管理、自动交易
"""
import time
import signal
import sys
from typing import Optional
from datetime import datetime
from loguru import logger
from .market_analyzer import MarketAnalyzer, MarketSignal
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .strategy_config import TradingConfig


class TradingEngine:
    """交易引擎"""

    def __init__(self, apex_exchange, config: TradingConfig):
        """
        初始化交易引擎

        Args:
            apex_exchange: Apex交易所实例
            config: 交易配置
        """
        self.apex = apex_exchange
        self.config = config
        self.running = False

        # 初始化各个模块
        self.market_analyzer = MarketAnalyzer(
            apex_exchange=apex_exchange,
            strategy_config=config.strategy,
            risk_config=config.risk,
        )

        self.position_manager = PositionManager(
            apex_exchange=apex_exchange,
            strategy_config=config.strategy,
            risk_config=config.risk,
        )

        self.risk_manager = RiskManager(
            apex_exchange=apex_exchange,
            position_manager=self.position_manager,
            risk_config=config.risk,
            dry_run=config.dry_run,
        )

        # 注册信号处理器（优雅退出）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("交易引擎初始化完成")

    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        logger.info(f"收到退出信号 {signum}，准备停止...")
        self.stop()

    def start(self):
        """启动交易引擎"""
        logger.info("=" * 60)
        logger.info("启动交易引擎")
        logger.info("=" * 60)
        logger.info(f"运行模式: {'模拟' if self.config.dry_run else '实盘'}")
        logger.info(f"交易对: {self.config.strategy.symbols}")
        logger.info(f"扫描间隔: {self.config.scan_interval}秒")
        logger.info("=" * 60)

        self.running = True

        # 加载之前的状态
        if self.config.save_state:
            self.position_manager.load_state(self.config.state_file)

        # 启动时首先同步持仓和检查风险
        logger.info("同步持仓...")
        self.position_manager.sync_positions()
        self._check_and_handle_existing_positions()

        # 主循环
        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"\n{'=' * 60}")
                logger.info(f"第 {iteration} 次扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'=' * 60}")

                # 1. 监控和处理风险
                self._monitor_risk()

                # 2. 扫描市场寻找交易机会
                self._scan_and_trade()

                # 3. 生成报告
                if iteration % 5 == 0:  # 每5次迭代生成一次报告
                    self._generate_report()

                # 4. 保存状态
                if self.config.save_state:
                    self.position_manager.save_state(self.config.state_file)

                # 等待下次扫描
                logger.info(f"等待 {self.config.scan_interval} 秒...")
                time.sleep(self.config.scan_interval)

            except KeyboardInterrupt:
                logger.info("收到中断信号，准备退出...")
                break
            except Exception as e:
                logger.error(f"主循环发生错误: {e}", exc_info=True)
                logger.info("等待60秒后重试...")
                time.sleep(60)

        logger.info("交易引擎已停止")

    def stop(self):
        """停止交易引擎"""
        self.running = False
        logger.info("正在停止交易引擎...")

        # 保存最终状态
        if self.config.save_state:
            self.position_manager.save_state(self.config.state_file)

        # 生成最终报告
        self._generate_report()

    def _check_and_handle_existing_positions(self):
        """检查和处理现有持仓"""
        positions = self.position_manager.get_all_positions()

        if not positions:
            logger.info("当前无持仓")
            return

        logger.info(f"检测到 {len(positions)} 个持仓:")
        for position in positions:
            logger.info(
                f"  - {position.symbol} {position.side} {position.size} @ {position.entry_price:.2f} "
                f"(当前价: {position.current_price:.2f}, 盈亏: {position.unrealized_pnl_pct:.2%})"
            )

            # 设置止损止盈（如果还没设置）
            if not position.stop_loss_price:
                self.position_manager._set_stop_loss_take_profit(position)

        # 立即执行一次风险检查
        logger.info("执行风险检查...")
        actions = self.risk_manager.monitor_positions()
        if actions:
            logger.info(f"风险控制操作: {actions}")

    def _monitor_risk(self):
        """监控和处理风险"""
        logger.info("监控风险...")
        actions = self.risk_manager.monitor_positions()

        if actions:
            logger.warning(f"执行风险控制操作: {actions}")

    def _scan_and_trade(self):
        """扫描市场并执行交易"""
        # 检查是否可以交易
        if self.position_manager._check_daily_loss_limit():
            logger.warning("达到每日最大亏损，停止扫描市场")
            return

        if self.position_manager._check_daily_trades_limit():
            logger.warning("达到每日最大交易次数，停止扫描市场")
            return

        # 扫描市场
        logger.info("扫描市场...")
        signals = self.market_analyzer.scan_market()

        if not signals:
            logger.info("未发现交易信号")
            return

        # 处理交易信号
        for signal in signals:
            self._handle_signal(signal)

    def _handle_signal(self, signal: MarketSignal):
        """
        处理交易信号

        Args:
            signal: 市场信号
        """
        logger.info(f"处理信号: {signal}")

        # 检查是否已有持仓
        if self.position_manager.has_position(signal.symbol):
            self._handle_signal_with_position(signal)
        else:
            self._handle_signal_without_position(signal)

    def _handle_signal_with_position(self, signal: MarketSignal):
        """
        当已有持仓时处理信号

        Args:
            signal: 市场信号
        """
        position = self.position_manager.get_position(signal.symbol)
        if not position:
            return

        logger.info(
            f"{signal.symbol} 已有持仓: {position.side} {position.size} "
            f"(盈亏: {position.unrealized_pnl_pct:.2%})"
        )

        # 检查是否应该平仓
        exit_signal = self.market_analyzer.check_exit_signal(signal.symbol, position.side)
        if exit_signal:
            logger.info(f"检测到平仓信号: {exit_signal.reason}")
            self.risk_manager._close_position(position, exit_signal.reason)
            return

        # 检查是否应该加仓或减仓
        if signal.signal_type == "BUY" and position.side == "LONG":
            if self.risk_manager.should_scale_in(position):
                logger.info(f"{signal.symbol} 考虑加多仓")
                # TODO: 实现加仓逻辑
        elif signal.signal_type == "SELL" and position.side == "SHORT":
            if self.risk_manager.should_scale_in(position):
                logger.info(f"{signal.symbol} 考虑加空仓")
                # TODO: 实现加仓逻辑

        if self.risk_manager.should_scale_out(position):
            logger.info(f"{signal.symbol} 考虑减仓")
            # TODO: 实现减仓逻辑

    def _handle_signal_without_position(self, signal: MarketSignal):
        """
        当无持仓时处理信号

        Args:
            signal: 市场信号
        """
        # 检查是否可以开新仓
        can_open, reason = self.position_manager.can_open_new_position(signal.symbol)
        if not can_open:
            logger.info(f"{signal.symbol} 无法开仓: {reason}")
            return

        # 计算仓位大小
        position_size = self.risk_manager.calculate_position_size(
            symbol=signal.symbol,
            price=signal.price,
            signal_strength=signal.strength,
        )

        if position_size <= 0:
            logger.warning(f"{signal.symbol} 计算仓位大小为0，跳过")
            return

        # 执行开仓
        self._open_position(signal, position_size)

    def _open_position(self, signal: MarketSignal, size: float):
        """
        开仓

        Args:
            signal: 市场信号
            size: 仓位大小
        """
        side = signal.signal_type  # "BUY" or "SELL"

        logger.info(f"准备开仓: {signal.symbol} {side} {size} @ {signal.price:.2f}")
        logger.info(f"信号强度: {signal.strength:.2f}, 原因: {signal.reason}")

        if self.config.dry_run:
            logger.info(f"[模拟模式] 开仓 {signal.symbol} {side} {size}")
            self.position_manager.record_trade()
            return

        # 实盘模式 - 限价单开仓
        try:
            order = self.apex.trade.create_order(
                symbol=signal.symbol,
                side=side,
                type="LIMIT",
                size=str(size),
                price=str(signal.price),
                time_in_force="GTC",
                post_only=True,  # 只做Maker
                client_order_id=f"open_{signal.symbol}_{int(time.time())}",
            )

            if order.get("success"):
                logger.info(f"开仓订单提交成功: {signal.symbol}")
                logger.info(f"订单详情: {order}")
                self.position_manager.record_trade()
            else:
                logger.error(f"开仓订单提交失败: {order}")

        except Exception as e:
            logger.error(f"开仓时发生错误: {e}")

    def _generate_report(self):
        """生成交易报告"""
        logger.info("\n" + "=" * 60)
        logger.info("交易报告")
        logger.info("=" * 60)

        report = self.risk_manager.get_risk_report()

        logger.info(f"账户余额: {report['账户余额']:.2f} USDC")
        logger.info(f"持仓数量: {report['持仓数量']}")
        logger.info(f"总持仓价值: {report['总持仓价值']:.2f} USDC")
        logger.info(f"总未实现盈亏: {report['总未实现盈亏']:.2f} USDC")
        logger.info(f"仓位占比: {report['仓位占比']:.2%}")
        logger.info(f"今日交易次数: {report['今日交易次数']}")
        logger.info(f"今日盈亏: {report['今日盈亏']:.2f} USDC")

        if report['持仓详情']:
            logger.info("\n持仓详情:")
            for pos in report['持仓详情']:
                logger.info(
                    f"  {pos['交易对']} {pos['方向']} {pos['数量']:.6f} - "
                    f"开仓价: {pos['开仓价']:.2f}, 当前价: {pos['当前价']:.2f}, "
                    f"盈亏: {pos['未实现盈亏']:.2f} ({pos['盈亏百分比']:.2%})"
                )
                if pos['保证金率']:
                    logger.info(f"    保证金率: {pos['保证金率']:.2%}")
                logger.info(
                    f"    止损: {pos['止损价']:.2f}, 止盈: {pos['止盈价']:.2f}, "
                    f"追踪止损: {pos['追踪止损价']:.2f if pos['追踪止损价'] else 'N/A'}"
                )

        logger.info("=" * 60 + "\n")

        # 自动评价
        self._auto_evaluate(report)

    def _auto_evaluate(self, report: Dict):
        """
        自动评价当前交易状态

        Args:
            report: 风险报告
        """
        logger.info("自动评价:")

        # 评价账户状态
        if report['仓位占比'] > self.config.risk.max_position_size * 0.8:
            logger.warning("  ⚠️  仓位占比较高，接近风险上限")
        elif report['仓位占比'] > self.config.risk.max_position_size * 0.5:
            logger.info("  ℹ️  仓位占比适中")
        else:
            logger.info("  ✓ 仓位占比保守")

        # 评价今日表现
        if report['今日盈亏'] > 0:
            logger.info(f"  ✓ 今日盈利 {report['今日盈亏']:.2f} USDC")
        elif report['今日盈亏'] < 0:
            loss_pct = abs(report['今日盈亏']) / report['账户余额'] if report['账户余额'] > 0 else 0
            if loss_pct > self.config.risk.max_daily_loss_pct * 0.8:
                logger.warning(f"  ⚠️  今日亏损接近限制: {loss_pct:.2%}")
            else:
                logger.info(f"  ℹ️  今日亏损 {report['今日盈亏']:.2f} USDC ({loss_pct:.2%})")

        # 评价交易频率
        trade_rate = report['今日交易次数'] / self.config.risk.max_daily_trades
        if trade_rate > 0.8:
            logger.warning(f"  ⚠️  今日交易次数接近限制: {report['今日交易次数']}/{self.config.risk.max_daily_trades}")
        elif trade_rate > 0.5:
            logger.info(f"  ℹ️  今日交易次数适中: {report['今日交易次数']}/{self.config.risk.max_daily_trades}")
        else:
            logger.info(f"  ✓ 今日交易次数保守: {report['今日交易次数']}/{self.config.risk.max_daily_trades}")

        # 评价持仓健康度
        for pos in report['持仓详情']:
            if pos['保证金率'] and pos['保证金率'] < self.config.risk.min_margin_ratio:
                logger.warning(
                    f"  ⚠️  {pos['交易对']} 保证金率偏低: {pos['保证金率']:.2%}"
                )

            if pos['盈亏百分比'] > 0.05:
                logger.info(f"  ✓ {pos['交易对']} 表现良好，盈利 {pos['盈亏百分比']:.2%}")
            elif pos['盈亏百分比'] < -0.02:
                logger.warning(f"  ⚠️  {pos['交易对']} 需要关注，亏损 {pos['盈亏百分比']:.2%}")

        logger.info("")

    def run_once(self):
        """单次运行（用于测试）"""
        logger.info("单次扫描模式")

        # 同步持仓
        self.position_manager.sync_positions()

        # 监控风险
        self._monitor_risk()

        # 扫描市场
        self._scan_and_trade()

        # 生成报告
        self._generate_report()
