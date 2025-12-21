#!/usr/bin/env python3
"""
快速启动脚本
自动化交易策略的便捷启动入口
"""
import sys
import os
from loguru import logger
from apex_trading_bot import ApexExchange
from strategy_config import create_safe_config, create_aggressive_config, TradingConfig
from trading_engine import TradingEngine


def setup_logging():
    """配置日志系统"""
    logger.remove()

    # 控制台输出 - 彩色格式
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # 创建logs目录
    os.makedirs("logs", exist_ok=True)

    # 文件输出 - 详细日志
    logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
    )

    logger.info("日志系统已初始化")


def get_config_from_env():
    """从环境变量获取API配置"""
    api_key = os.getenv("APEX_API_KEY", "")
    secret_key = os.getenv("APEX_SECRET_KEY", "")
    passphrase = os.getenv("APEX_PASSPHRASE", "")
    testnet = os.getenv("APEX_TESTNET", "true").lower() == "true"

    return api_key, secret_key, passphrase, testnet


def main():
    """主函数"""
    setup_logging()

    logger.info("=" * 80)
    logger.info("Apex Trading Bot - 自动化交易系统")
    logger.info("=" * 80)

    # 从环境变量或用户输入获取配置
    api_key, secret_key, passphrase, testnet = get_config_from_env()

    if not all([api_key, secret_key, passphrase]):
        logger.warning("未设置环境变量，使用交互式配置")
        print("\n请输入API配置:")
        api_key = input("API Key: ").strip()
        secret_key = input("Secret Key: ").strip()
        passphrase = input("Passphrase: ").strip()
        testnet_input = input("使用测试网? (y/n, 默认y): ").strip().lower()
        testnet = testnet_input != "n"

    # 选择配置模式
    print("\n请选择运行模式:")
    print("1. 安全保守模式（推荐新手）- 低风险，小仓位")
    print("2. 激进模式（有经验者）- 高风险高收益")
    print("3. 自定义模式")

    choice = input("\n选择 (1-3, 默认1): ").strip() or "1"

    if choice == "1":
        config = create_safe_config()
        logger.info("使用安全保守配置")
    elif choice == "2":
        config = create_aggressive_config()
        logger.warning("使用激进配置 - 请注意风险！")
    else:
        logger.info("使用自定义配置")
        config = TradingConfig()

    # 设置API配置
    config.api_key = api_key
    config.secret_key = secret_key
    config.passphrase = passphrase
    config.testnet = testnet

    # 设置运行模式
    dry_run_input = input("\n模拟模式运行? (y/n, 默认y): ").strip().lower()
    config.dry_run = dry_run_input != "n"

    # 确认配置
    logger.info("\n配置总结:")
    logger.info(f"  网络: {'测试网' if config.testnet else '主网'}")
    logger.info(f"  模式: {'模拟' if config.dry_run else '实盘'}")
    logger.info(f"  交易对: {config.strategy.symbols}")
    logger.info(f"  时间框架: {config.strategy.timeframe}")
    logger.info(f"  最大仓位: {config.risk.max_position_size * 100:.0f}%")
    logger.info(f"  止损: {config.risk.stop_loss_pct * 100:.1f}%")
    logger.info(f"  止盈: {config.risk.take_profit_pct * 100:.1f}%")
    logger.info(f"  每日最大亏损: {config.risk.max_daily_loss_pct * 100:.1f}%")

    if not config.dry_run:
        logger.warning("\n警告: 即将在实盘模式运行!")
        confirm = input("确认继续? (yes/no): ").strip().lower()
        if confirm != "yes":
            logger.info("已取消运行")
            return

    # 创建交易所实例并启动引擎
    try:
        logger.info("\n连接到Apex Exchange...")
        with ApexExchange(
            api_key=config.api_key,
            secret_key=config.secret_key,
            passphrase=config.passphrase,
            testnet=config.testnet,
        ) as apex:
            logger.info("连接成功!")

            # 测试连接
            try:
                time_data = apex.public.get_system_time()
                if time_data.get("success"):
                    logger.info(f"服务器时间: {time_data}")
                else:
                    logger.error("无法获取服务器时间")
                    return
            except Exception as e:
                logger.error(f"连接测试失败: {e}")
                return

            # 创建交易引擎
            logger.info("\n初始化交易引擎...")
            engine = TradingEngine(apex, config)

            # 启动交易
            logger.info("\n启动交易引擎...")
            logger.info("按 Ctrl+C 停止\n")

            try:
                engine.start()
            except KeyboardInterrupt:
                logger.info("\n收到停止信号")
                engine.stop()

    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n程序已退出")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"未处理的错误: {e}", exc_info=True)
        sys.exit(1)
