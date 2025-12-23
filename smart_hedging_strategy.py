"""
智能Delta中性对冲策略

特点:
1. 只设定方向和目标仓位大小，系统自动维护
2. 根据资金费率动态选择做多/做空的币种
3. 根据价格波动自动调仓保持Delta中性
4. 波动率自适应杠杆 (高波动低杠杆，低波动高杠杆)
"""
import os
import time
import json
import logging
import urllib.request
import ssl
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SmartHedgingConfig:
    """智能对冲策略配置"""

    # 交易的币种列表
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT", "SOL-USDT"])

    # 目标仓位 (单边，多头或空头的总仓位)
    target_notional_per_side: float = 50000.0  # 多头50K + 空头50K = 总敞口100K

    # 杠杆范围 (根据波动率动态调整)
    min_leverage: int = 2
    max_leverage: int = 5
    base_leverage: int = 3

    # Delta中性阈值
    delta_threshold_pct: float = 0.02  # 2% 偏差触发调仓
    emergency_delta_pct: float = 0.05  # 5% 紧急调仓

    # 调仓参数
    min_trade_notional: float = 100.0  # 最小交易额
    max_single_trade_pct: float = 0.10  # 单次最大调整10%仓位

    # 波动率计算窗口 (小时)
    volatility_window_hours: int = 24

    # 检查间隔
    check_interval_seconds: int = 30  # 30秒检查一次
    funding_check_interval_seconds: int = 300  # 5分钟检查一次费率


@dataclass
class AssetState:
    """资产状态"""
    symbol: str
    side: Optional[PositionSide] = None  # 当前方向
    target_notional: float = 0.0  # 目标仓位
    current_quantity: float = 0.0  # 当前持仓数量
    current_price: float = 0.0  # 当前价格
    funding_rate: float = 0.0  # 当前资金费率
    predicted_funding_rate: float = 0.0  # 预测资金费率
    volatility_24h: float = 0.0  # 24小时波动率

    @property
    def current_notional(self) -> float:
        """当前仓位价值"""
        return abs(self.current_quantity) * self.current_price

    @property
    def position_delta(self) -> float:
        """仓位Delta (多头正，空头负)"""
        if self.side == PositionSide.LONG:
            return self.current_notional
        elif self.side == PositionSide.SHORT:
            return -self.current_notional
        return 0.0


class SmartHedgingStrategy:
    """智能对冲策略"""

    def __init__(self, config: SmartHedgingConfig):
        self.config = config
        self.assets: Dict[str, AssetState] = {}
        self.current_leverage = config.base_leverage
        self.last_funding_check = 0
        self.last_volatility_check = 0
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(timestamp, price)]

        # 初始化资产状态
        for symbol in config.symbols:
            self.assets[symbol] = AssetState(symbol=symbol)
            self.price_history[symbol] = []

    def update_price(self, symbol: str, price: float):
        """更新价格并记录历史"""
        if symbol in self.assets:
            self.assets[symbol].current_price = price

            # 记录价格历史
            now = time.time()
            self.price_history[symbol].append((now, price))

            # 清理超过窗口期的历史数据
            cutoff = now - self.config.volatility_window_hours * 3600
            self.price_history[symbol] = [
                (t, p) for t, p in self.price_history[symbol] if t > cutoff
            ]

    def update_position(self, symbol: str, quantity: float, side: Optional[PositionSide]):
        """更新持仓信息"""
        if symbol in self.assets:
            self.assets[symbol].current_quantity = abs(quantity)
            self.assets[symbol].side = side

    def update_funding_rates(self, rates: Dict[str, dict]):
        """更新资金费率信息"""
        for symbol, info in rates.items():
            if symbol in self.assets:
                self.assets[symbol].funding_rate = info.get("rate", 0)
                self.assets[symbol].predicted_funding_rate = info.get("predicted_rate", 0)

    def calculate_volatility(self, symbol: str) -> float:
        """计算波动率 (基于价格历史的标准差)"""
        history = self.price_history.get(symbol, [])
        if len(history) < 10:  # 数据不足
            return 0.05  # 默认5%

        prices = [p for _, p in history]

        # 计算收益率
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        if not returns:
            return 0.05

        # 计算标准差
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance)

        # 年化波动率 (假设数据点间隔约30秒)
        # 一天约2880个30秒间隔
        annualized = std * math.sqrt(2880 * 365)

        return min(annualized, 2.0)  # 上限200%

    def calculate_optimal_leverage(self) -> int:
        """根据波动率计算最优杠杆"""
        # 计算所有币种的平均波动率
        volatilities = []
        for symbol in self.assets:
            vol = self.calculate_volatility(symbol)
            self.assets[symbol].volatility_24h = vol
            volatilities.append(vol)

        if not volatilities:
            return self.config.base_leverage

        avg_vol = sum(volatilities) / len(volatilities)

        # 波动率与杠杆的反向关系
        # 低波动(< 30%年化) -> 高杠杆
        # 高波动(> 80%年化) -> 低杠杆
        if avg_vol < 0.30:
            leverage = self.config.max_leverage
        elif avg_vol > 0.80:
            leverage = self.config.min_leverage
        else:
            # 线性插值
            ratio = (avg_vol - 0.30) / (0.80 - 0.30)
            leverage = int(self.config.max_leverage - ratio * (self.config.max_leverage - self.config.min_leverage))

        leverage = max(self.config.min_leverage, min(self.config.max_leverage, leverage))

        logger.info(f"[波动率] 平均: {avg_vol*100:.1f}% -> 杠杆: {leverage}x")
        for symbol, asset in self.assets.items():
            logger.info(f"  {symbol}: {asset.volatility_24h*100:.1f}%")

        return leverage

    def calculate_optimal_positions(self) -> Dict[str, Tuple[PositionSide, float]]:
        """
        根据资金费率计算最优持仓配置

        策略:
        - 正费率最高的币做空 (空头收费率)
        - 费率最低/最负的币做多 (多头收费率或少付)
        - 保持多空平衡

        Returns:
            {symbol: (side, target_notional)}
        """
        # 按资金费率排序 (高到低)
        sorted_assets = sorted(
            self.assets.items(),
            key=lambda x: x[1].funding_rate,
            reverse=True
        )

        target_per_side = self.config.target_notional_per_side
        result = {}

        if len(sorted_assets) == 3:
            # 3个币: 最高费率做空，最低费率做多
            # 中间币根据费率正负决定
            highest = sorted_assets[0]
            middle = sorted_assets[1]
            lowest = sorted_assets[2]

            # 最低费率做多 (全部多头仓位)
            result[lowest[0]] = (PositionSide.LONG, target_per_side)

            # 最高和中间做空，各分一半空头仓位
            if middle[1].funding_rate > 0:
                # 中间也是正费率，空头各一半
                result[highest[0]] = (PositionSide.SHORT, target_per_side / 2)
                result[middle[0]] = (PositionSide.SHORT, target_per_side / 2)
            else:
                # 中间是负费率，只空最高的
                result[highest[0]] = (PositionSide.SHORT, target_per_side)
                result[middle[0]] = (PositionSide.LONG, 0)  # 不持仓
                # 调整多头
                result[lowest[0]] = (PositionSide.LONG, target_per_side / 2)
                result[middle[0]] = (PositionSide.LONG, target_per_side / 2)

        elif len(sorted_assets) == 2:
            # 2个币: 高费率空，低费率多
            highest = sorted_assets[0]
            lowest = sorted_assets[1]
            result[highest[0]] = (PositionSide.SHORT, target_per_side)
            result[lowest[0]] = (PositionSide.LONG, target_per_side)

        elif len(sorted_assets) == 1:
            # 单币无法对冲
            logger.warning("只有一个币种，无法构建Delta中性策略")
            return {}

        return result

    def get_portfolio_delta(self) -> Tuple[float, float, float]:
        """
        获取组合Delta

        Returns:
            (total_long, total_short, net_delta)
        """
        total_long = 0.0
        total_short = 0.0

        for asset in self.assets.values():
            if asset.side == PositionSide.LONG:
                total_long += asset.current_notional
            elif asset.side == PositionSide.SHORT:
                total_short += asset.current_notional

        net_delta = total_long - total_short
        return total_long, total_short, net_delta

    def check_rebalance_needed(self) -> Tuple[bool, str]:
        """检查是否需要调仓"""
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short

        if total_exposure < self.config.min_trade_notional:
            return True, "需要建仓"

        imbalance_pct = abs(net_delta) / total_exposure if total_exposure > 0 else 0

        # 紧急调仓
        if imbalance_pct >= self.config.emergency_delta_pct:
            return True, f"紧急调仓: Delta偏差 {imbalance_pct:.1%}"

        # 常规调仓
        if imbalance_pct >= self.config.delta_threshold_pct:
            return True, f"Delta偏差 {imbalance_pct:.1%} >= {self.config.delta_threshold_pct:.1%}"

        # 检查是否需要向目标仓位调整
        optimal = self.calculate_optimal_positions()
        for symbol, (target_side, target_notional) in optimal.items():
            asset = self.assets.get(symbol)
            if not asset:
                continue

            # 方向不对
            if asset.side != target_side and asset.current_notional > self.config.min_trade_notional:
                return True, f"{symbol} 方向需要调整: {asset.side} -> {target_side}"

            # 仓位偏差过大
            if target_notional > 0:
                diff_pct = abs(asset.current_notional - target_notional) / target_notional
                if diff_pct > 0.10:  # 10% 偏差
                    return True, f"{symbol} 仓位偏差 {diff_pct:.1%}"

        return False, "无需调仓"

    def calculate_rebalance_actions(self) -> List[dict]:
        """
        计算调仓操作

        Returns:
            [{"symbol": str, "side": str, "quantity": float, "notional": float, "reason": str}]
        """
        actions = []
        optimal = self.calculate_optimal_positions()
        total_long, total_short, net_delta = self.get_portfolio_delta()

        for symbol, (target_side, target_notional) in optimal.items():
            asset = self.assets.get(symbol)
            if not asset or asset.current_price <= 0:
                continue

            current_notional = asset.current_notional
            current_side = asset.side

            # 情况1: 需要平仓并反向开仓
            if current_side and current_side != target_side and current_notional > self.config.min_trade_notional:
                # 先平掉当前仓位
                close_side = "SELL" if current_side == PositionSide.LONG else "BUY"
                actions.append({
                    "symbol": symbol,
                    "side": close_side,
                    "quantity": asset.current_quantity,
                    "notional": current_notional,
                    "reason": f"平仓 {current_side.value}",
                    "reduce_only": True
                })
                # 然后开新仓
                if target_notional > 0:
                    open_side = "BUY" if target_side == PositionSide.LONG else "SELL"
                    open_qty = target_notional / asset.current_price
                    actions.append({
                        "symbol": symbol,
                        "side": open_side,
                        "quantity": open_qty,
                        "notional": target_notional,
                        "reason": f"开仓 {target_side.value}"
                    })

            # 情况2: 方向正确，调整仓位大小
            elif current_side == target_side or current_notional < self.config.min_trade_notional:
                diff = target_notional - current_notional

                # 限制单次调整幅度
                max_adjust = self.config.target_notional_per_side * self.config.max_single_trade_pct
                diff = max(-max_adjust, min(max_adjust, diff))

                if abs(diff) >= self.config.min_trade_notional:
                    if diff > 0:
                        # 加仓
                        side = "BUY" if target_side == PositionSide.LONG else "SELL"
                        qty = diff / asset.current_price
                        actions.append({
                            "symbol": symbol,
                            "side": side,
                            "quantity": qty,
                            "notional": diff,
                            "reason": f"加仓 {target_side.value} +${diff:.0f}"
                        })
                    else:
                        # 减仓
                        side = "SELL" if target_side == PositionSide.LONG else "BUY"
                        qty = abs(diff) / asset.current_price
                        actions.append({
                            "symbol": symbol,
                            "side": side,
                            "quantity": qty,
                            "notional": abs(diff),
                            "reason": f"减仓 {target_side.value} -${abs(diff):.0f}",
                            "reduce_only": True
                        })

        # 验证操作后的Delta平衡
        # (简化：实际应该模拟执行后检查)

        return actions

    def get_status(self) -> dict:
        """获取策略状态"""
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short
        imbalance_pct = abs(net_delta) / total_exposure if total_exposure > 0 else 0

        return {
            "total_long": total_long,
            "total_short": total_short,
            "net_delta": net_delta,
            "imbalance_pct": imbalance_pct,
            "is_neutral": imbalance_pct < self.config.delta_threshold_pct,
            "leverage": self.current_leverage,
            "assets": {
                symbol: {
                    "side": asset.side.value if asset.side else None,
                    "current_notional": asset.current_notional,
                    "target_notional": asset.target_notional,
                    "funding_rate": asset.funding_rate,
                    "volatility_24h": asset.volatility_24h
                }
                for symbol, asset in self.assets.items()
            }
        }


# ============ 数据获取函数 ============

def fetch_binance_price(symbol: str) -> Optional[float]:
    """从Binance获取价格"""
    binance_symbol = symbol.replace("-", "")
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"

    ssl_ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return float(data.get("price", 0))
    except Exception as e:
        logger.error(f"获取Binance价格失败 {symbol}: {e}")
        return None


def fetch_apex_funding_info(symbol: str, testnet: bool = False) -> Optional[dict]:
    """从Apex获取资金费率信息"""
    endpoint = "https://testnet.omni.apex.exchange" if testnet else "https://omni.apex.exchange"
    ticker_symbol = symbol.replace("-", "")
    url = f"{endpoint}/api/v3/ticker?symbol={ticker_symbol}"

    ssl_ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
            tickers = data.get("data", [])
            if tickers:
                ticker = tickers[0]
                rate = ticker.get("fundingRate")
                predicted = ticker.get("predictedFundingRate")
                next_time_str = ticker.get("nextFundingTime", "")

                if rate is not None:
                    result = {
                        "rate": float(rate),
                        "predicted_rate": float(predicted) if predicted else 0.0,
                        "next_funding_time": next_time_str
                    }
                    return result
    except Exception as e:
        logger.error(f"获取Apex资金费率失败 {symbol}: {e}")
    return None


def fetch_all_funding_info(symbols: List[str], testnet: bool = False) -> Dict[str, dict]:
    """获取多个币种的资金费率"""
    result = {}
    for symbol in symbols:
        info = fetch_apex_funding_info(symbol, testnet)
        if info:
            result[symbol] = info
    return result


# ============ 主程序示例 ============

def main():
    """测试策略"""
    config = SmartHedgingConfig(
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        target_notional_per_side=50000.0,
        min_leverage=2,
        max_leverage=5,
        base_leverage=3,
        delta_threshold_pct=0.02,
        check_interval_seconds=30
    )

    strategy = SmartHedgingStrategy(config)

    print("=" * 60)
    print("智能Delta中性对冲策略")
    print("=" * 60)
    print(f"目标仓位: 多头 ${config.target_notional_per_side:,.0f} + 空头 ${config.target_notional_per_side:,.0f}")
    print(f"杠杆范围: {config.min_leverage}x - {config.max_leverage}x")
    print(f"Delta阈值: {config.delta_threshold_pct:.0%}")
    print("=" * 60)

    # 获取初始数据
    print("\n获取市场数据...")

    # 更新价格
    for symbol in config.symbols:
        price = fetch_binance_price(symbol)
        if price:
            strategy.update_price(symbol, price)
            print(f"  {symbol}: ${price:,.2f}")

    # 更新资金费率
    print("\n获取资金费率...")
    funding_info = fetch_all_funding_info(config.symbols, testnet=False)
    strategy.update_funding_rates(funding_info)

    for symbol, info in sorted(funding_info.items(), key=lambda x: x[1]["rate"], reverse=True):
        direction = "SHORT有利" if info["rate"] > 0 else "LONG有利"
        print(f"  {symbol}: {info['rate']*100:+.6f}% ({direction})")

    # 计算最优配置
    print("\n计算最优持仓配置...")
    optimal = strategy.calculate_optimal_positions()
    for symbol, (side, notional) in optimal.items():
        if notional > 0:
            print(f"  {symbol}: {side.value} ${notional:,.0f}")

    # 计算杠杆
    print("\n计算最优杠杆...")
    leverage = strategy.calculate_optimal_leverage()
    strategy.current_leverage = leverage

    # 获取状态
    print("\n策略状态:")
    status = strategy.get_status()
    print(f"  多头总计: ${status['total_long']:,.0f}")
    print(f"  空头总计: ${status['total_short']:,.0f}")
    print(f"  净Delta: ${status['net_delta']:,.0f}")
    print(f"  Delta偏差: {status['imbalance_pct']:.2%}")
    print(f"  是否中性: {'是' if status['is_neutral'] else '否'}")
    print(f"  当前杠杆: {status['leverage']}x")


if __name__ == "__main__":
    main()
