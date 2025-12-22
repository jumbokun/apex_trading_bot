"""
策略系统测试脚本
用于验证各个模块是否正常工作
"""
import sys
from loguru import logger
from apex_trading_bot import ApexExchange
from strategy_config import create_safe_config
from market_analyzer import MarketAnalyzer, TechnicalIndicators
from position_manager import PositionManager, Position
from risk_manager import RiskManager


def setup_test_logging():
    """配置测试日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG",
    )


def test_technical_indicators():
    """测试技术指标计算"""
    logger.info("=" * 60)
    logger.info("测试1: 技术指标计算")
    logger.info("=" * 60)

    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 108, 111, 113, 112]
    highs = [101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 111, 109, 112, 114, 113]
    lows = [99, 101, 100, 102, 104, 103, 105, 107, 106, 108, 109, 107, 110, 112, 111]

    indicators = TechnicalIndicators()

    sma = indicators.calculate_sma(prices, 10)
    ema = indicators.calculate_ema(prices, 10)
    rsi = indicators.calculate_rsi(prices, 14)
    volatility = indicators.calculate_volatility(prices, 10)
    atr = indicators.calculate_atr(highs, lows, prices, 10)

    logger.info(f"SMA(10): {sma:.2f}")
    logger.info(f"EMA(10): {ema:.2f}")
    logger.info(f"RSI(14): {rsi:.2f}")
    logger.info(f"Volatility: {volatility:.4f}")
    logger.info(f"ATR(10): {atr:.2f}")

    assert sma is not None, "SMA计算失败"
    assert ema is not None, "EMA计算失败"
    assert rsi is not None, "RSI计算失败"
    assert volatility is not None, "Volatility计算失败"
    assert atr is not None, "ATR计算失败"

    logger.success("✓ 技术指标计算测试通过")
    return True


def test_position_management():
    """测试仓位管理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 仓位管理")
    logger.info("=" * 60)

    # 创建模拟持仓
    position = Position(
        symbol="BTC-USDC",
        side="LONG",
        size=0.01,
        entry_price=100000.0,
        current_price=100000.0,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
        margin=1000.0,
        leverage=3.0,
        liquidation_price=95000.0,
    )

    logger.info(f"创建持仓: {position.symbol} {position.side} {position.size}")

    # 测试价格更新
    position.update_price(105000.0)
    logger.info(f"价格更新到 {position.current_price:.2f}")
    logger.info(f"未实现盈亏: {position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2%})")

    assert position.unrealized_pnl > 0, "盈利计算错误"

    # 测试止损检查
    position.stop_loss_price = 98000.0
    position.take_profit_price = 110000.0

    logger.info(f"止损价: {position.stop_loss_price:.2f}")
    logger.info(f"止盈价: {position.take_profit_price:.2f}")

    assert not position.should_stop_loss(), "不应该触发止损"
    assert not position.should_take_profit(), "不应该触发止盈"

    # 测试追踪止损
    position.update_trailing_stop(0.03)
    logger.info(f"追踪止损价: {position.trailing_stop_price:.2f}")

    logger.success("✓ 仓位管理测试通过")
    return True


def test_market_analyzer_connection():
    """测试市场分析器（需要网络连接）"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 市场分析器（需要网络）")
    logger.info("=" * 60)

    config = create_safe_config()

    try:
        with ApexExchange(
            api_key="",
            secret_key="",
            passphrase="",
            testnet=True
        ) as apex:
            # 测试公共API连接
            logger.info("测试API连接...")
            time_data = apex.public.get_system_time()

            if not time_data.get("success"):
                logger.warning("API连接失败，跳过市场分析器测试")
                return False

            logger.info(f"服务器时间: {time_data.get('data')}")

            # 创建市场分析器
            analyzer = MarketAnalyzer(apex, config.strategy, config.risk)

            # 测试单个交易对分析
            logger.info("分析 BTC-USDC...")
            signal = analyzer._analyze_symbol("BTC-USDC")

            if signal:
                logger.info(f"生成信号: {signal}")
                logger.info(f"  信号类型: {signal.signal_type}")
                logger.info(f"  信号强度: {signal.strength:.2f}")
                logger.info(f"  价格: {signal.price:.2f}")
                logger.info(f"  原因: {signal.reason}")
            else:
                logger.info("当前无明确信号")

            logger.success("✓ 市场分析器测试通过")
            return True

    except Exception as e:
        logger.error(f"市场分析器测试失败: {e}")
        return False


def test_risk_calculations():
    """测试风险计算"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 风险计算")
    logger.info("=" * 60)

    # 模拟不同的保证金率场景
    scenarios = [
        {"margin": 1000, "unrealized_pnl": 200, "position_value": 3000, "name": "健康"},
        {"margin": 1000, "unrealized_pnl": -500, "position_value": 3000, "name": "警告"},
        {"margin": 1000, "unrealized_pnl": -700, "position_value": 3000, "name": "危险"},
    ]

    for scenario in scenarios:
        margin = scenario["margin"]
        pnl = scenario["unrealized_pnl"]
        value = scenario["position_value"]

        margin_ratio = (margin + pnl) / value if value > 0 else 0

        logger.info(f"\n场景: {scenario['name']}")
        logger.info(f"  保证金: {margin:.2f}")
        logger.info(f"  未实现盈亏: {pnl:.2f}")
        logger.info(f"  持仓价值: {value:.2f}")
        logger.info(f"  保证金率: {margin_ratio:.2%}")

        if margin_ratio < 0.15:
            logger.warning(f"  ⚠️  保证金率过低!")
        elif margin_ratio < 0.30:
            logger.info(f"  ℹ️  保证金率偏低")
        else:
            logger.info(f"  ✓ 保证金率健康")

    logger.success("✓ 风险计算测试通过")
    return True


def test_config_presets():
    """测试配置预设"""
    logger.info("\n" + "=" * 60)
    logger.info("测试5: 配置预设")
    logger.info("=" * 60)

    from strategy_config import create_safe_config, create_aggressive_config

    safe = create_safe_config()
    aggressive = create_aggressive_config()

    logger.info("安全配置:")
    logger.info(f"  最大杠杆: {safe.risk.max_leverage}x")
    logger.info(f"  最大仓位: {safe.risk.max_position_size * 100:.0f}%")
    logger.info(f"  止损: {safe.risk.stop_loss_pct * 100:.1f}%")

    logger.info("\n激进配置:")
    logger.info(f"  最大杠杆: {aggressive.risk.max_leverage}x")
    logger.info(f"  最大仓位: {aggressive.risk.max_position_size * 100:.0f}%")
    logger.info(f"  止损: {aggressive.risk.stop_loss_pct * 100:.1f}%")

    assert safe.risk.max_leverage < aggressive.risk.max_leverage, "配置预设错误"

    logger.success("✓ 配置预设测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    setup_test_logging()

    logger.info("开始运行策略系统测试...")
    logger.info("")

    results = {
        "技术指标计算": test_technical_indicators(),
        "仓位管理": test_position_management(),
        "市场分析器": test_market_analyzer_connection(),
        "风险计算": test_risk_calculations(),
        "配置预设": test_config_presets(),
    }

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)

    logger.info("")
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过: {passed_tests}")
    logger.info(f"失败: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        logger.success("\n✓ 所有测试通过! 系统可以正常使用。")
        return True
    else:
        logger.warning(f"\n⚠️  {total_tests - passed_tests} 个测试失败，请检查配置。")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n测试被中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试出错: {e}", exc_info=True)
        sys.exit(1)
