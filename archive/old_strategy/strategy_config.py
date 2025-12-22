"""
策略配置文件
定义交易策略的所有参数和风险控制规则
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RiskConfig:
    """风险控制配置"""

    # 仓位控制
    max_position_size: float = 0.1  # 最大仓位占账户资金的比例 (10%)
    max_leverage: float = 3.0  # 最大杠杆倍数

    # 止损止盈
    stop_loss_pct: float = 0.02  # 固定止损百分比 (2%)
    take_profit_pct: float = 0.05  # 止盈百分比 (5%)
    trailing_stop_pct: float = 0.03  # 追踪止损百分比 (3%)

    # 爆仓保护 - 当保证金率低于此值时强制平仓
    min_margin_ratio: float = 0.15  # 最小保证金率 (15%)，远高于交易所爆仓线
    emergency_close_margin_ratio: float = 0.20  # 紧急平仓保证金率 (20%)

    # 每日风险限制
    max_daily_loss_pct: float = 0.05  # 每日最大亏损 (5%)
    max_daily_trades: int = 20  # 每日最大交易次数

    # 市场波动性过滤
    max_volatility: float = 0.10  # 最大允许的市场波动率 (10%)
    min_volume_ratio: float = 0.5  # 最小成交量比率（相对于平均成交量）

    # 资金费率限制
    max_funding_rate: float = 0.0005  # 最大资金费率 (0.05%)


@dataclass
class StrategyConfig:
    """交易策略配置"""

    # 交易对
    symbols: List[str] = None  # 例如: ["BTC-USDC", "ETH-USDC"]

    # 时间框架
    timeframe: str = "15m"  # K线周期: 1m, 5m, 15m, 30m, 1h, 4h, 1d
    kline_lookback: int = 100  # 回看K线数量

    # 技术指标参数
    ma_short_period: int = 10  # 短期均线周期
    ma_long_period: int = 30  # 长期均线周期
    rsi_period: int = 14  # RSI周期
    rsi_overbought: float = 70.0  # RSI超买阈值
    rsi_oversold: float = 30.0  # RSI超卖阈值

    # 成交量分析
    volume_ma_period: int = 20  # 成交量均线周期
    volume_surge_ratio: float = 1.5  # 成交量激增比率

    # 信号确认
    min_signal_strength: float = 0.6  # 最小信号强度 (0-1)

    # 仓位管理
    initial_position_pct: float = 0.03  # 初始仓位占资金的比例 (3%)
    scale_in_pct: float = 0.02  # 加仓比例 (2%)
    scale_out_pct: float = 0.5  # 减仓比例 (50% of position)

    # 交易时间控制
    avoid_funding_time: bool = True  # 是否避开资金费率结算时间
    trading_hours: Optional[List[tuple]] = None  # 交易时间段 [(0, 24)] 表示全天

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC-USDC"]
        if self.trading_hours is None:
            self.trading_hours = [(0, 24)]  # 默认全天交易


@dataclass
class TradingConfig:
    """总体交易配置"""

    # API配置
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    testnet: bool = True

    # 策略和风险配置
    strategy: StrategyConfig = None
    risk: RiskConfig = None

    # 运行模式
    dry_run: bool = True  # 模拟运行模式（不实际下单）
    log_level: str = "INFO"  # 日志级别

    # 扫描间隔
    scan_interval: int = 60  # 市场扫描间隔（秒）

    # 持久化
    save_state: bool = True  # 是否保存状态
    state_file: str = "trading_state.json"  # 状态文件路径

    def __post_init__(self):
        if self.strategy is None:
            self.strategy = StrategyConfig()
        if self.risk is None:
            self.risk = RiskConfig()


def load_default_config() -> TradingConfig:
    """加载默认配置"""
    return TradingConfig(
        strategy=StrategyConfig(
            symbols=["BTC-USDC"],
            timeframe="15m",
        ),
        risk=RiskConfig(),
        dry_run=True,
        testnet=True,
    )


def create_safe_config() -> TradingConfig:
    """
    创建安全配置（适合初学者和保守交易者）
    特点:
    - 低杠杆
    - 严格止损
    - 小仓位
    - 低频交易
    """
    return TradingConfig(
        strategy=StrategyConfig(
            symbols=["BTC-USDC"],
            timeframe="1h",  # 使用较长时间框架
            initial_position_pct=0.02,  # 更小的初始仓位
            scale_in_pct=0.01,
            min_signal_strength=0.7,  # 更高的信号要求
        ),
        risk=RiskConfig(
            max_position_size=0.05,  # 最大5%仓位
            max_leverage=2.0,  # 最大2倍杠杆
            stop_loss_pct=0.015,  # 1.5%止损
            take_profit_pct=0.04,  # 4%止盈
            max_daily_loss_pct=0.03,  # 每日最大3%亏损
            max_daily_trades=10,  # 每日最多10笔交易
        ),
        dry_run=True,
        testnet=True,
    )


def create_aggressive_config() -> TradingConfig:
    """
    创建激进配置（适合经验丰富的交易者）
    警告: 高风险高收益
    """
    return TradingConfig(
        strategy=StrategyConfig(
            symbols=["BTC-USDC", "ETH-USDC"],
            timeframe="5m",  # 更短时间框架
            initial_position_pct=0.05,
            scale_in_pct=0.03,
            min_signal_strength=0.5,
        ),
        risk=RiskConfig(
            max_position_size=0.15,  # 最大15%仓位
            max_leverage=5.0,  # 最大5倍杠杆
            stop_loss_pct=0.03,
            take_profit_pct=0.08,
            max_daily_loss_pct=0.08,
            max_daily_trades=30,
        ),
        dry_run=True,
        testnet=True,
    )
