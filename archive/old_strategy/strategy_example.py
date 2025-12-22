"""
交易策略使用示例
展示如何使用自动化交易系统
"""
import sys
from loguru import logger
from apex_trading_bot import ApexExchange
from strategy_config import TradingConfig, StrategyConfig, RiskConfig, create_safe_config
from trading_engine import TradingEngine


def setup_logging():
    """配置日志"""
    logger.remove()  # 移除默认处理器

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # 添加文件输出
    logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="00:00",  # 每天轮换
        retention="30 days",  # 保留30天
        compression="zip",  # 压缩旧日志
    )


def example_safe_trading():
    """
    示例1: 安全保守的交易策略
    - 低杠杆（2倍）
    - 小仓位（最大5%）
    - 严格止损（1.5%）
    - 适合初学者
    """
    logger.info("=" * 80)
    logger.info("示例1: 安全保守的交易策略")
    logger.info("=" * 80)

    # 创建安全配置
    config = create_safe_config()

    # 填入你的API密钥
    config.api_key = "your_api_key_here"
    config.secret_key = "your_secret_key_here"
    config.passphrase = "your_passphrase_here"
    config.testnet = True  # 使用测试网
    config.dry_run = True  # 模拟模式

    # 自定义策略参数
    config.strategy.symbols = ["BTC-USDC"]
    config.strategy.timeframe = "1h"  # 1小时K线
    config.scan_interval = 300  # 每5分钟扫描一次

    # 创建交易所实例
    with ApexExchange(
        api_key=config.api_key,
        secret_key=config.secret_key,
        passphrase=config.passphrase,
        testnet=config.testnet,
    ) as apex:
        # 创建交易引擎
        engine = TradingEngine(apex, config)

        # 启动交易（会持续运行直到手动停止）
        engine.start()


def example_custom_strategy():
    """
    示例2: 自定义交易策略
    根据你的需求定制参数
    """
    logger.info("=" * 80)
    logger.info("示例2: 自定义交易策略")
    logger.info("=" * 80)

    # 创建自定义配置
    config = TradingConfig(
        api_key="your_api_key_here",
        secret_key="your_secret_key_here",
        passphrase="your_passphrase_here",
        testnet=True,
        dry_run=True,

        strategy=StrategyConfig(
            # 交易对列表
            symbols=["BTC-USDC", "ETH-USDC"],

            # K线周期和回看数量
            timeframe="15m",
            kline_lookback=100,

            # 技术指标参数
            ma_short_period=10,
            ma_long_period=30,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,

            # 成交量分析
            volume_ma_period=20,
            volume_surge_ratio=1.5,

            # 信号强度要求
            min_signal_strength=0.65,

            # 仓位管理
            initial_position_pct=0.03,  # 初始仓位3%
            scale_in_pct=0.02,  # 加仓2%
            scale_out_pct=0.5,  # 减仓50%
        ),

        risk=RiskConfig(
            # 仓位和杠杆限制
            max_position_size=0.10,  # 最大10%仓位
            max_leverage=3.0,  # 最大3倍杠杆

            # 止损止盈
            stop_loss_pct=0.02,  # 2%止损
            take_profit_pct=0.05,  # 5%止盈
            trailing_stop_pct=0.03,  # 3%追踪止损

            # 爆仓保护
            min_margin_ratio=0.15,  # 15%最低保证金率
            emergency_close_margin_ratio=0.20,  # 20%紧急平仓

            # 每日限制
            max_daily_loss_pct=0.05,  # 每日最大5%亏损
            max_daily_trades=20,  # 每日最多20笔

            # 市场过滤
            max_volatility=0.10,  # 最大10%波动率
            min_volume_ratio=0.5,  # 最小50%成交量
            max_funding_rate=0.0005,  # 最大0.05%资金费率
        ),

        scan_interval=60,  # 每60秒扫描一次
        save_state=True,  # 保存状态
        state_file="my_trading_state.json",
    )

    # 创建交易所实例
    with ApexExchange(
        api_key=config.api_key,
        secret_key=config.secret_key,
        passphrase=config.passphrase,
        testnet=config.testnet,
    ) as apex:
        # 创建交易引擎
        engine = TradingEngine(apex, config)

        # 启动交易
        engine.start()


def example_single_scan():
    """
    示例3: 单次扫描模式
    只运行一次，不持续监控
    适合测试或手动触发
    """
    logger.info("=" * 80)
    logger.info("示例3: 单次扫描模式")
    logger.info("=" * 80)

    config = create_safe_config()
    config.api_key = "your_api_key_here"
    config.secret_key = "your_secret_key_here"
    config.passphrase = "your_passphrase_here"
    config.testnet = True
    config.dry_run = True

    with ApexExchange(
        api_key=config.api_key,
        secret_key=config.secret_key,
        passphrase=config.passphrase,
        testnet=config.testnet,
    ) as apex:
        engine = TradingEngine(apex, config)

        # 只运行一次
        engine.run_once()


def example_risk_check_only():
    """
    示例4: 仅风险检查模式
    不执行新交易，只监控现有仓位的风险
    适合睡觉前运行，确保现有仓位安全
    """
    logger.info("=" * 80)
    logger.info("示例4: 仅风险检查模式")
    logger.info("=" * 80)

    config = create_safe_config()
    config.api_key = "your_api_key_here"
    config.secret_key = "your_secret_key_here"
    config.passphrase = "your_passphrase_here"
    config.testnet = True
    config.dry_run = False  # 实盘模式，可以真实平仓

    with ApexExchange(
        api_key=config.api_key,
        secret_key=config.secret_key,
        passphrase=config.passphrase,
        testnet=config.testnet,
    ) as apex:
        from risk_manager import RiskManager
        from position_manager import PositionManager

        # 只创建风险管理模块
        position_manager = PositionManager(apex, config.strategy, config.risk)
        risk_manager = RiskManager(apex, position_manager, config.risk, dry_run=config.dry_run)

        # 同步并检查现有持仓
        position_manager.sync_positions()

        # 生成风险报告
        report = risk_manager.get_risk_report()

        logger.info("风险报告:")
        logger.info(f"  账户余额: {report['账户余额']:.2f} USDC")
        logger.info(f"  持仓数量: {report['持仓数量']}")
        logger.info(f"  总未实现盈亏: {report['总未实现盈亏']:.2f} USDC")

        # 执行风险监控（会自动执行止损止盈）
        actions = risk_manager.monitor_positions()
        if actions:
            logger.warning(f"执行了风险控制操作: {actions}")
        else:
            logger.info("所有持仓健康，无需操作")


if __name__ == "__main__":
    # 设置日志
    setup_logging()

    # 选择要运行的示例
    logger.info("请选择要运行的示例:")
    logger.info("1. 安全保守策略（持续运行）")
    logger.info("2. 自定义策略（持续运行）")
    logger.info("3. 单次扫描")
    logger.info("4. 仅风险检查")

    choice = input("\n输入选项 (1-4): ").strip()

    if choice == "1":
        example_safe_trading()
    elif choice == "2":
        example_custom_strategy()
    elif choice == "3":
        example_single_scan()
    elif choice == "4":
        example_risk_check_only()
    else:
        logger.error("无效选项")
        sys.exit(1)
