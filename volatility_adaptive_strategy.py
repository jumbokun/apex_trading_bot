"""
波动率自适应Delta中性策略 - Volatility Adaptive Neutral Strategy
================================================================

核心逻辑：
1. 根据市场波动率自动调整目标仓位大小
   - 高波动率 -> 降低仓位（风险控制）
   - 低波动率 -> 增加仓位（捕捉更多收益）

2. 动态分配空头仓位
   - ETH和SOL按各自波动率反比分配
   - 高波动的币种占比更少

3. 保持Delta中性
   - 多头总价值 ≈ 空头总价值
   - 自动调仓维持平衡

波动率计算：
- 使用24小时价格波动幅度 (High-Low)/Close
- 或使用ATR (Average True Range)

仓位公式：
- target_notional = max_notional - (volatility - vol_low) / (vol_high - vol_low) * (max_notional - min_notional)
- 当 volatility <= vol_low 时，target = max_notional
- 当 volatility >= vol_high 时，target = min_notional
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import math
import json
import ssl
import urllib.request
from datetime import datetime, timedelta


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class VolatilityConfig:
    """波动率配置

    使用 ATR (Average True Range) 计算波动率
    - ATR 是业界标准的波动率指标 (J. Welles Wilder Jr.)
    - 14 周期是最常用的设置
    - ATR% = ATR / 收盘价 * 100

    典型的加密货币 ATR% 范围:
    - 低波动: < 2% (横盘/震荡市)
    - 中等波动: 2-4% (正常市场)
    - 高波动: > 4% (趋势/事件驱动)
    """
    # ATR% 阈值
    vol_low: float = 0.015   # 1.5% - 低波动阈值 (满仓)
    vol_high: float = 0.04   # 4% - 高波动阈值 (最低仓)

    # ATR周期 (业界标准14周期)
    atr_period: int = 14

    # 使用的K线周期
    kline_interval: str = "1h"  # 1小时K线


@dataclass
class AdaptiveHedgingConfig:
    """波动率自适应对冲配置"""

    # 仓位上下限 (单边)
    min_notional: float = 10000.0   # 最小单边仓位 10,000U
    max_notional: float = 45000.0   # 最大单边仓位 45,000U

    # 币种配置
    long_symbol: str = "BTC-USDT"   # 多头币种
    short_symbols: List[str] = field(default_factory=lambda: ["ETH-USDT", "SOL-USDT"])

    # 波动率配置
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)

    # 杠杆
    leverage: int = 3

    # 调仓参数
    rebalance_interval_seconds: int = 300   # 5分钟检查
    min_rebalance_interval_seconds: int = 300
    scale_amount: float = 2000.0           # 每次调仓金额
    min_trade_notional: float = 100.0

    # Delta中性阈值
    delta_threshold_pct: float = 0.005     # 0.5%偏差触发 (约$150 @ 30k单边)
    emergency_delta_pct: float = 0.03      # 3%紧急调仓

    # 波动率检查间隔
    volatility_check_interval: int = 300   # 5分钟检查一次波动率


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    quantity: float = 0.0
    avg_price: float = 0.0
    target_notional: float = 0.0
    current_volatility: float = 0.0  # 当前波动率

    def get_notional(self, price: float) -> float:
        return self.quantity * price

    def get_delta(self, price: float) -> float:
        notional = self.get_notional(price)
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


class VolatilityCalculator:
    """波动率计算器 - 使用 ATR (Average True Range)

    ATR 是 J. Welles Wilder Jr. 发明的标准波动率指标。
    比简单的 (high-low)/close 更准确，因为考虑了跳空缺口。
    """

    def __init__(self):
        self.price_history: Dict[str, List[dict]] = {}
        # cache: key -> (atr, atr_pct, timestamp)
        self.volatility_cache: Dict[str, Tuple[float, float, float]] = {}
        self.cache_ttl = 60  # 缓存60秒

    def fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 24) -> List[dict]:
        """从Binance获取K线数据"""
        binance_symbol = symbol.replace("-", "")
        url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"

        try:
            ssl_ctx = ssl.create_default_context()
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                klines = []
                for k in data:
                    klines.append({
                        'time': k[0],
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
                return klines
        except Exception as e:
            print(f"获取K线失败 {symbol}: {e}")
            return []

    def calculate_atr(self, symbol: str, period: int = 14, interval: str = "1h") -> Tuple[float, float]:
        """计算 ATR (Average True Range) - 业界标准波动率指标

        ATR 由 J. Welles Wilder Jr. 发明，是衡量市场波动性的标准指标。
        True Range = max(高-低, |高-前收|, |低-前收|)
        ATR = True Range 的移动平均

        Args:
            symbol: 交易对
            period: ATR周期 (默认14，业界标准)
            interval: K线周期

        Returns:
            (ATR绝对值, ATR百分比)
        """
        # 检查缓存
        now = time.time()
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.volatility_cache:
            atr, atr_pct, ts = self.volatility_cache[cache_key]
            if now - ts < self.cache_ttl:
                return atr, atr_pct

        # 获取 K 线数据 (需要 period+1 根来计算 period 个 TR)
        klines = self.fetch_klines(symbol, interval, period + 1)
        if len(klines) < period + 1:
            return 0.0, 0.05

        # 计算 True Range
        true_ranges = []
        for i in range(1, len(klines)):
            high = klines[i]['high']
            low = klines[i]['low']
            prev_close = klines[i-1]['close']

            # True Range = max(高-低, |高-前收|, |低-前收|)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # ATR = TR 的简单移动平均
        atr = sum(true_ranges[-period:]) / period

        # ATR% = ATR / 当前价格
        current_price = klines[-1]['close']
        atr_pct = atr / current_price if current_price > 0 else 0.05

        # 缓存结果
        self.volatility_cache[cache_key] = (atr, atr_pct, now)

        return atr, atr_pct

    def calculate_volatility(self, symbol: str, period: int = 14, interval: str = "1h") -> float:
        """计算波动率 (ATR%)

        返回 ATR 百分比，用于仓位计算
        """
        _, atr_pct = self.calculate_atr(symbol, period, interval)
        return atr_pct

    # 向后兼容的别名
    def calculate_24h_volatility(self, symbol: str) -> float:
        """计算波动率 (使用ATR14)"""
        return self.calculate_volatility(symbol, period=14)

    def get_volatility_score(self, symbol: str, config: VolatilityConfig) -> float:
        """获取波动率评分 (0-1)

        0 = 最低波动 (vol <= vol_low)
        1 = 最高波动 (vol >= vol_high)
        """
        vol = self.calculate_24h_volatility(symbol)

        if vol <= config.vol_low:
            return 0.0
        elif vol >= config.vol_high:
            return 1.0
        else:
            return (vol - config.vol_low) / (config.vol_high - config.vol_low)


class VolatilityAdaptiveStrategy:
    """波动率自适应Delta中性策略"""

    def __init__(self, config: AdaptiveHedgingConfig = None):
        self.config = config or AdaptiveHedgingConfig()
        self.vol_calc = VolatilityCalculator()

        self.positions: Dict[str, Position] = {}
        self.prices: Dict[str, float] = {}
        self.volatilities: Dict[str, float] = {}

        self.last_volatility_check = 0
        self.last_rebalance_time = 0
        self.rebalance_count = 0
        self.total_volume = 0.0

        # 初始化持仓
        self._init_positions()

    def _init_positions(self):
        """初始化持仓结构"""
        # 多头
        self.positions[self.config.long_symbol] = Position(
            symbol=self.config.long_symbol,
            side=PositionSide.LONG,
            target_notional=self.config.max_notional
        )

        # 空头
        for symbol in self.config.short_symbols:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.SHORT,
                target_notional=self.config.max_notional / len(self.config.short_symbols)
            )

    def update_price(self, symbol: str, price: float):
        """更新价格"""
        self.prices[symbol] = price

    def update_position(self, symbol: str, quantity: float, avg_price: float):
        """更新持仓"""
        if symbol in self.positions:
            self.positions[symbol].quantity = quantity
            self.positions[symbol].avg_price = avg_price

    def update_volatility(self, force: bool = False) -> Dict[str, float]:
        """更新波动率数据

        Returns:
            {symbol: volatility} 字典
        """
        now = time.time()

        # 检查是否需要更新
        if not force and now - self.last_volatility_check < self.config.volatility_check_interval:
            return self.volatilities

        self.last_volatility_check = now

        # 计算所有币种波动率
        all_symbols = [self.config.long_symbol] + self.config.short_symbols
        for symbol in all_symbols:
            vol = self.vol_calc.calculate_24h_volatility(symbol)
            self.volatilities[symbol] = vol
            if symbol in self.positions:
                self.positions[symbol].current_volatility = vol

        return self.volatilities

    def calculate_target_notional(self) -> Tuple[float, Dict[str, float]]:
        """根据波动率计算目标仓位

        Returns:
            (总目标仓位, {symbol: 目标仓位})
        """
        # 更新波动率
        self.update_volatility()

        # 计算整体市场波动率 (取BTC波动率作为基准)
        btc_vol = self.volatilities.get(self.config.long_symbol, 0.05)
        vol_score = self.vol_calc.get_volatility_score(
            self.config.long_symbol,
            self.config.volatility
        )

        # 根据波动率计算目标仓位
        # 高波动 (score=1) -> min_notional
        # 低波动 (score=0) -> max_notional
        target_per_side = self.config.max_notional - vol_score * (
            self.config.max_notional - self.config.min_notional
        )

        targets = {}

        # 多头目标
        targets[self.config.long_symbol] = target_per_side

        # 空头目标 - 按波动率反比分配
        short_vols = {}
        for symbol in self.config.short_symbols:
            short_vols[symbol] = self.volatilities.get(symbol, 0.05)

        # 计算反比权重
        # 波动率越高，权重越低
        total_inverse_vol = sum(1/v for v in short_vols.values() if v > 0)

        for symbol in self.config.short_symbols:
            vol = short_vols.get(symbol, 0.05)
            if vol > 0 and total_inverse_vol > 0:
                # 反比权重
                weight = (1/vol) / total_inverse_vol
            else:
                # 平均分配
                weight = 1.0 / len(self.config.short_symbols)

            targets[symbol] = target_per_side * weight

        return target_per_side * 2, targets  # 总仓位 = 多头 + 空头

    def get_portfolio_delta(self) -> Tuple[float, float, float]:
        """计算组合Delta

        Returns: (总多头, 总空头, 净Delta)
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

        return total_long, total_short, total_long - total_short

    def should_rebalance(self) -> Tuple[bool, str]:
        """判断是否需要调仓"""
        now = time.time()

        # 检查时间间隔
        if now - self.last_rebalance_time < self.config.min_rebalance_interval_seconds:
            remaining = self.config.min_rebalance_interval_seconds - (now - self.last_rebalance_time)
            return False, f"冷却中 (还需{remaining:.0f}秒)"

        # 计算目标仓位
        total_target, targets = self.calculate_target_notional()

        # 检查是否有仓位需要调整
        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            if price <= 0:
                continue

            current = pos.get_notional(price)
            target = targets.get(symbol, 0)
            gap = abs(target - current)

            # 需要加仓或减仓
            if gap >= self.config.min_trade_notional:
                direction = "加仓" if target > current else "减仓"
                return True, f"{symbol} 需要{direction} (当前${current:.0f} -> 目标${target:.0f})"

        # 检查Delta偏差
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short

        if total_exposure > 0:
            imbalance = abs(net_delta) / total_exposure
            if imbalance >= self.config.emergency_delta_pct:
                return True, f"紧急: Delta偏差 {imbalance:.1%}"
            if imbalance >= self.config.delta_threshold_pct:
                return True, f"Delta偏差 {imbalance:.1%}"

        return False, "无需调仓"

    def calculate_rebalance_actions(self) -> List[RebalanceAction]:
        """计算调仓操作"""
        actions = []

        # 获取目标仓位
        total_target, targets = self.calculate_target_notional()

        # 计算每个币种需要的调整
        adjustments = []  # [(symbol, current, target, gap, price)]

        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            if price <= 0:
                continue

            current = pos.get_notional(price)
            target = targets.get(symbol, 0)
            gap = target - current  # 正数=需要加仓, 负数=需要减仓

            if abs(gap) >= self.config.min_trade_notional:
                adjustments.append((symbol, pos, current, target, gap, price))

        if not adjustments:
            return actions

        # 按gap大小排序（优先处理偏差大的）
        adjustments.sort(key=lambda x: abs(x[4]), reverse=True)

        # 分离加仓和减仓
        scale_up = [(s, p, c, t, g, pr) for s, p, c, t, g, pr in adjustments if g > 0]
        scale_down = [(s, p, c, t, g, pr) for s, p, c, t, g, pr in adjustments if g < 0]

        # 限制每次调仓金额
        max_adjust = self.config.scale_amount

        # 处理需要加仓的
        long_scale_up = [x for x in scale_up if x[1].side == PositionSide.LONG]
        short_scale_up = [x for x in scale_up if x[1].side == PositionSide.SHORT]

        # 处理需要减仓的
        long_scale_down = [x for x in scale_down if x[1].side == PositionSide.LONG]
        short_scale_down = [x for x in scale_down if x[1].side == PositionSide.SHORT]

        # Delta中性：多空同时加仓或同时减仓
        if long_scale_up and short_scale_up:
            # 同时加多头和空头
            long_item = long_scale_up[0]

            # 多头加仓金额
            long_add = min(max_adjust / 2, long_item[4])
            if long_add >= self.config.min_trade_notional:
                qty = long_add / long_item[5]
                actions.append(RebalanceAction(
                    symbol=long_item[0],
                    side="BUY",
                    quantity=qty,
                    notional=long_add,
                    reason=f"加仓 LONG (波动率调整)"
                ))

            # 空头加仓 - 按比例分配给多个空头
            short_add_total = long_add
            for short_item in short_scale_up:
                # 按gap比例分配
                total_short_gap = sum(x[4] for x in short_scale_up)
                if total_short_gap > 0:
                    ratio = short_item[4] / total_short_gap
                else:
                    ratio = 1.0 / len(short_scale_up)

                short_add = short_add_total * ratio
                if short_add >= self.config.min_trade_notional / 2:
                    qty = short_add / short_item[5]
                    actions.append(RebalanceAction(
                        symbol=short_item[0],
                        side="SELL",
                        quantity=qty,
                        notional=short_add,
                        reason=f"加仓 SHORT (波动率调整)"
                    ))

        elif long_scale_down and short_scale_down:
            # 同时减多头和空头
            long_item = long_scale_down[0]

            # 多头减仓金额
            long_reduce = min(max_adjust / 2, abs(long_item[4]))
            if long_reduce >= self.config.min_trade_notional:
                qty = long_reduce / long_item[5]
                actions.append(RebalanceAction(
                    symbol=long_item[0],
                    side="SELL",
                    quantity=qty,
                    notional=long_reduce,
                    reason=f"减仓 LONG (波动率调整)"
                ))

            # 空头减仓
            short_reduce_total = long_reduce
            for short_item in short_scale_down:
                total_short_gap = sum(abs(x[4]) for x in short_scale_down)
                if total_short_gap > 0:
                    ratio = abs(short_item[4]) / total_short_gap
                else:
                    ratio = 1.0 / len(short_scale_down)

                short_reduce = short_reduce_total * ratio
                if short_reduce >= self.config.min_trade_notional / 2:
                    qty = short_reduce / short_item[5]
                    actions.append(RebalanceAction(
                        symbol=short_item[0],
                        side="BUY",
                        quantity=qty,
                        notional=short_reduce,
                        reason=f"减仓 SHORT (波动率调整)"
                    ))

        # 处理单边调整 (只有一边需要调整)
        # 允许空头之间重新分配 (一个加一个减)
        elif short_scale_up and short_scale_down:
            # 空头内部重新平衡 (例如 ETH 加仓 + SOL 减仓)
            for up_item in short_scale_up:
                add = min(max_adjust / 2, up_item[4])
                if add >= self.config.min_trade_notional:
                    qty = add / up_item[5]
                    actions.append(RebalanceAction(
                        symbol=up_item[0],
                        side="SELL",
                        quantity=qty,
                        notional=add,
                        reason=f"空头重平衡 (加仓)"
                    ))
            for down_item in short_scale_down:
                reduce = min(max_adjust / 2, abs(down_item[4]))
                if reduce >= self.config.min_trade_notional:
                    qty = reduce / down_item[5]
                    actions.append(RebalanceAction(
                        symbol=down_item[0],
                        side="BUY",
                        quantity=qty,
                        notional=reduce,
                        reason=f"空头重平衡 (减仓)"
                    ))

        elif long_scale_up and not short_scale_up:
            # 只有多头需要加，检查是否偏空
            total_long, total_short, net_delta = self.get_portfolio_delta()
            if net_delta < 0:  # 偏空，可以加多头
                long_item = long_scale_up[0]
                add = min(max_adjust / 2, abs(net_delta), long_item[4])
                if add >= self.config.min_trade_notional:
                    qty = add / long_item[5]
                    actions.append(RebalanceAction(
                        symbol=long_item[0],
                        side="BUY",
                        quantity=qty,
                        notional=add,
                        reason=f"Delta修正 (偏空)"
                    ))

        elif short_scale_up and not long_scale_up:
            # 只有空头需要加，检查是否偏多
            total_long, total_short, net_delta = self.get_portfolio_delta()
            if net_delta > 0:  # 偏多，可以加空头
                for short_item in short_scale_up:
                    add = min(max_adjust / 4, net_delta / len(short_scale_up), short_item[4])
                    if add >= self.config.min_trade_notional / 2:
                        qty = add / short_item[5]
                        actions.append(RebalanceAction(
                            symbol=short_item[0],
                            side="SELL",
                            quantity=qty,
                            notional=add,
                            reason=f"Delta修正 (偏多)"
                        ))

        return actions

    def record_rebalance(self, actions: List[RebalanceAction]):
        """记录调仓"""
        self.last_rebalance_time = time.time()
        self.rebalance_count += 1
        for action in actions:
            self.total_volume += action.notional

    def get_status(self) -> dict:
        """获取策略状态"""
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short
        total_target, targets = self.calculate_target_notional()

        positions_detail = []
        for symbol, pos in self.positions.items():
            price = self.prices.get(symbol, 0)
            target = targets.get(symbol, 0)
            positions_detail.append({
                "symbol": symbol,
                "side": pos.side.value,
                "quantity": pos.quantity,
                "price": price,
                "current_notional": pos.get_notional(price),
                "target_notional": target,
                "volatility": self.volatilities.get(symbol, 0),
                "fill_pct": pos.get_notional(price) / target * 100 if target > 0 else 0
            })

        return {
            "total_long": total_long,
            "total_short": total_short,
            "net_delta": net_delta,
            "total_exposure": total_exposure,
            "target_exposure": total_target,
            "imbalance_pct": abs(net_delta) / total_exposure if total_exposure > 0 else 0,
            "rebalance_count": self.rebalance_count,
            "total_volume": self.total_volume,
            "positions": positions_detail,
            "volatilities": self.volatilities
        }


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("波动率自适应策略 - ATR 测试")
    print("=" * 60)

    config = AdaptiveHedgingConfig(
        min_notional=10000.0,
        max_notional=50000.0,
        long_symbol="BTC-USDT",
        short_symbols=["ETH-USDT", "SOL-USDT"]
    )

    strategy = VolatilityAdaptiveStrategy(config)

    # 模拟价格
    strategy.update_price("BTC-USDT", 95000.0)
    strategy.update_price("ETH-USDT", 3300.0)
    strategy.update_price("SOL-USDT", 180.0)

    # 测试 ATR 计算
    print("\n正在计算 ATR(14) 波动率...")
    print("-" * 40)

    vol_calc = strategy.vol_calc
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    for symbol in symbols:
        atr, atr_pct = vol_calc.calculate_atr(symbol, period=14, interval="1h")
        price = strategy.prices.get(symbol, 0)
        print(f"{symbol}:")
        print(f"  价格: ${price:,.0f}")
        print(f"  ATR(14): ${atr:,.2f}")
        print(f"  ATR%: {atr_pct*100:.2f}%")

    # 更新波动率
    print("\n" + "-" * 40)
    vols = strategy.update_volatility(force=True)

    # 波动率阈值
    print(f"\n波动率阈值配置:")
    print(f"  低波动 (满仓): < {config.volatility.vol_low*100:.1f}%")
    print(f"  高波动 (最低仓): > {config.volatility.vol_high*100:.1f}%")

    # 波动率评分
    print(f"\n波动率评分:")
    for symbol in symbols:
        vol = vols.get(symbol, 0)
        score = vol_calc.get_volatility_score(symbol, config.volatility)
        print(f"  {symbol}: ATR% {vol*100:.2f}% -> 评分 {score:.2f}")

    # 计算目标仓位
    total, targets = strategy.calculate_target_notional()
    print(f"\n目标仓位 (总计 ${total:,.0f}):")
    for symbol, target in targets.items():
        pos = strategy.positions[symbol]
        print(f"  {symbol}: {pos.side.value} ${target:,.0f}")

    # 检查是否需要调仓
    should, reason = strategy.should_rebalance()
    print(f"\n需要调仓: {should} - {reason}")

    if should:
        actions = strategy.calculate_rebalance_actions()
        print(f"\n调仓操作:")
        for action in actions:
            print(f"  {action}")
