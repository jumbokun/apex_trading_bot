"""
自适应Delta中性对冲策略

特点:
1. 不固定方向 - 完全根据资金费率动态决定多空
2. 只设定目标仓位大小，系统自动选择最优方向
3. 费率变化时缓慢调仓（分批限价单）
4. Delta偏移时自动调仓保持中性
5. 波动率自适应杠杆
"""
import os
import sys
import time
import json
import logging
import urllib.request
import ssl
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

# 尝试导入 apexomni SDK
try:
    from apexomni import HttpPrivate
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    HttpPrivate = None

from dotenv import load_dotenv
load_dotenv()

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
class AdaptiveConfig:
    """自适应对冲配置"""

    # 交易的币种
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT", "SOL-USDT"])

    # 目标仓位（单边）- 这是最大上限
    max_notional_per_side: float = 50000.0  # 用户设定的最大值
    min_notional_per_side: float = 10000.0  # 最小仓位（极高波动时）

    # 杠杆范围 (Delta中性策略风险较低，可用高杠杆)
    min_leverage: int = 20   # 极端波动时降到20x
    max_leverage: int = 50   # 低波动时用50x
    base_leverage: int = 30  # 正常情况30x

    # 波动率参数
    volatility_window_hours: int = 24  # 波动率计算窗口
    low_volatility_threshold: float = 0.02   # 2% 日波动率视为低波动
    high_volatility_threshold: float = 0.05  # 5% 日波动率视为高波动
    extreme_volatility_threshold: float = 0.08  # 8% 日波动率视为极端 (更保守)

    # Delta中性阈值
    delta_threshold_pct: float = 0.02  # 2% 触发调仓
    emergency_delta_pct: float = 0.05  # 5% 紧急调仓

    # 费率变化阈值（触发方向调整）
    funding_rate_change_threshold: float = 0.0005  # 0.05% 费率变化触发检查

    # 调仓参数
    min_trade_notional: float = 100.0
    max_single_trade_pct: float = 0.05  # 单次最大调整5%仓位
    rebalance_batch_count: int = 10  # 大调仓分10批完成
    rebalance_batch_interval_seconds: int = 60  # 每批间隔60秒

    # 限价单参数
    limit_order_offset_pct: float = 0.0005  # 0.05% 挂单偏移
    order_timeout_seconds: int = 30  # 挂单超时
    max_retry_count: int = 3  # 最大重试次数

    # 检查间隔
    check_interval_seconds: int = 30
    funding_check_interval_seconds: int = 300  # 5分钟检查费率


@dataclass
class PendingRebalance:
    """待执行的调仓任务"""
    symbol: str
    from_side: Optional[PositionSide]
    to_side: PositionSide
    total_notional: float  # 总调仓金额
    remaining_notional: float  # 剩余调仓金额
    batches_completed: int = 0
    started_at: float = 0
    reason: str = ""


class AdaptiveHedgingStrategy:
    """自适应对冲策略"""

    def __init__(self, config: AdaptiveConfig):
        self.config = config

        # 当前状态
        self.current_positions: Dict[str, dict] = {}  # {symbol: {side, quantity, notional}}
        self.current_prices: Dict[str, float] = {}
        self.current_funding_rates: Dict[str, float] = {}
        self.previous_funding_rates: Dict[str, float] = {}

        # 价格历史（用于计算波动率）
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # {symbol: [(timestamp, price)]}
        self.volatility: Dict[str, float] = {}  # {symbol: volatility}
        self.avg_volatility: float = 0.0  # 平均波动率

        # 自适应参数（根据波动率动态调整）
        self.current_leverage: int = config.base_leverage
        self.current_target_notional: float = config.max_notional_per_side

        # 目标配置（根据费率动态计算）
        self.target_positions: Dict[str, dict] = {}  # {symbol: {side, notional}}

        # 待执行的调仓任务
        self.pending_rebalances: List[PendingRebalance] = []

        # 统计
        self.total_volume = 0.0
        self.trades_executed = 0

    def update_price(self, symbol: str, price: float):
        """更新价格并记录历史"""
        self.current_prices[symbol] = price

        # 记录价格历史
        current_time = time.time()
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append((current_time, price))

        # 清理过期数据（保留窗口期内的数据）
        window_seconds = self.config.volatility_window_hours * 3600
        cutoff_time = current_time - window_seconds
        self.price_history[symbol] = [
            (t, p) for t, p in self.price_history[symbol]
            if t >= cutoff_time
        ]

    def calculate_volatility(self, symbol: str) -> float:
        """
        计算单个币种的波动率

        使用简单的最高价-最低价/平均价 作为波动率指标
        """
        history = self.price_history.get(symbol, [])

        if len(history) < 10:  # 数据不足
            return 0.0

        prices = [p for _, p in history]
        high = max(prices)
        low = min(prices)
        avg = sum(prices) / len(prices)

        if avg <= 0:
            return 0.0

        # 波动率 = (最高-最低) / 平均价
        volatility = (high - low) / avg
        return volatility

    def update_volatility(self):
        """更新所有币种的波动率"""
        volatilities = []

        for symbol in self.config.symbols:
            vol = self.calculate_volatility(symbol)
            self.volatility[symbol] = vol
            if vol > 0:
                volatilities.append(vol)

        # 计算平均波动率
        if volatilities:
            self.avg_volatility = sum(volatilities) / len(volatilities)
        else:
            self.avg_volatility = 0.0

        return self.avg_volatility

    def get_adaptive_leverage(self) -> int:
        """
        根据波动率计算自适应杠杆

        规则:
        - 低波动 (<2%): 使用最大杠杆 (更多收益)
        - 中波动 (2%-5%): 使用基础杠杆
        - 高波动 (5%-10%): 降低杠杆
        - 极端波动 (>10%): 使用最小杠杆 (保护本金)
        """
        vol = self.avg_volatility

        if vol <= 0:
            return self.config.base_leverage

        if vol <= self.config.low_volatility_threshold:
            # 低波动 -> 高杠杆
            leverage = self.config.max_leverage
        elif vol <= self.config.high_volatility_threshold:
            # 中波动 -> 线性插值
            ratio = (vol - self.config.low_volatility_threshold) / \
                    (self.config.high_volatility_threshold - self.config.low_volatility_threshold)
            leverage = int(self.config.max_leverage - ratio * (self.config.max_leverage - self.config.base_leverage))
        elif vol <= self.config.extreme_volatility_threshold:
            # 高波动 -> 进一步降低
            ratio = (vol - self.config.high_volatility_threshold) / \
                    (self.config.extreme_volatility_threshold - self.config.high_volatility_threshold)
            leverage = int(self.config.base_leverage - ratio * (self.config.base_leverage - self.config.min_leverage))
        else:
            # 极端波动 -> 最小杠杆
            leverage = self.config.min_leverage

        return max(self.config.min_leverage, min(leverage, self.config.max_leverage))

    def get_adaptive_target_notional(self) -> float:
        """
        根据波动率计算自适应目标仓位

        规则 (假设 max=50000, min=10000):
        - 低波动 (<2%):  100% = $50,000
        - 中波动 (2%-5%): 60%-100% = $30,000-$50,000
        - 高波动 (5%-8%): 20%-60% = $10,000-$30,000
        - 极端波动 (>8%): 20% = $10,000 (最小)
        """
        vol = self.avg_volatility

        if vol <= 0:
            return self.config.max_notional_per_side

        max_notional = self.config.max_notional_per_side
        min_notional = self.config.min_notional_per_side

        # 计算最小仓位占比
        min_ratio = min_notional / max_notional  # 例如 10000/50000 = 0.2 (20%)

        if vol <= self.config.low_volatility_threshold:
            # 低波动 -> 满仓 (100%)
            target = max_notional
        elif vol <= self.config.high_volatility_threshold:
            # 中波动 -> 线性从100%降到60%
            ratio = (vol - self.config.low_volatility_threshold) / \
                    (self.config.high_volatility_threshold - self.config.low_volatility_threshold)
            pct = 1.0 - ratio * 0.4  # 100% -> 60%
            target = max_notional * pct
        elif vol <= self.config.extreme_volatility_threshold:
            # 高波动 -> 线性从60%降到最小(20%)
            ratio = (vol - self.config.high_volatility_threshold) / \
                    (self.config.extreme_volatility_threshold - self.config.high_volatility_threshold)
            pct = 0.6 - ratio * (0.6 - min_ratio)  # 60% -> 20%
            target = max_notional * pct
        else:
            # 极端波动 -> 最小仓位
            target = min_notional

        return max(min_notional, min(target, max_notional))

    def update_adaptive_params(self):
        """更新自适应参数（杠杆和目标仓位）"""
        old_leverage = self.current_leverage
        old_target = self.current_target_notional

        # 更新波动率
        self.update_volatility()

        # 计算新的自适应参数
        self.current_leverage = self.get_adaptive_leverage()
        self.current_target_notional = self.get_adaptive_target_notional()

        # 记录变化
        if self.current_leverage != old_leverage:
            logger.info(f"[自适应杠杆] {old_leverage}x -> {self.current_leverage}x (波动率: {self.avg_volatility*100:.2f}%)")

        if abs(self.current_target_notional - old_target) > 1000:
            logger.info(f"[自适应仓位] ${old_target:,.0f} -> ${self.current_target_notional:,.0f} (波动率: {self.avg_volatility*100:.2f}%)")

        return self.current_leverage, self.current_target_notional

    def update_position(self, symbol: str, quantity: float, side: Optional[PositionSide]):
        """更新持仓"""
        price = self.current_prices.get(symbol, 0)
        notional = abs(quantity) * price if price > 0 else 0

        self.current_positions[symbol] = {
            "side": side,
            "quantity": abs(quantity),
            "notional": notional
        }

    def update_funding_rates(self, rates: Dict[str, float]):
        """更新资金费率"""
        self.previous_funding_rates = self.current_funding_rates.copy()
        self.current_funding_rates = rates

    def calculate_optimal_targets(self) -> Dict[str, dict]:
        """
        根据资金费率计算最优目标持仓

        策略:
        - 费率最高的币做空（收费率）
        - 费率最低的币做多（少付/收费率）
        - 保持Delta中性
        - 使用自适应目标仓位（根据波动率调整）
        """
        if not self.current_funding_rates:
            return {}

        # 按费率排序（高到低）
        sorted_rates = sorted(
            self.current_funding_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 使用自适应目标仓位（根据波动率动态调整）
        target = self.current_target_notional
        targets = {}

        if len(sorted_rates) >= 3:
            highest = sorted_rates[0]  # 费率最高 -> 空
            middle = sorted_rates[1]
            lowest = sorted_rates[2]   # 费率最低 -> 多

            # 最低费率做多（全部多头仓位）
            targets[lowest[0]] = {"side": PositionSide.LONG, "notional": target}

            # 最高和中间做空
            if middle[1] > 0:
                # 中间也是正费率，两个空头各一半
                targets[highest[0]] = {"side": PositionSide.SHORT, "notional": target / 2}
                targets[middle[0]] = {"side": PositionSide.SHORT, "notional": target / 2}
            else:
                # 中间是负/零费率，只空最高的
                targets[highest[0]] = {"side": PositionSide.SHORT, "notional": target}
                # 中间币不持仓或也做多
                targets[middle[0]] = {"side": PositionSide.LONG, "notional": 0}

        elif len(sorted_rates) == 2:
            highest = sorted_rates[0]
            lowest = sorted_rates[1]
            targets[highest[0]] = {"side": PositionSide.SHORT, "notional": target}
            targets[lowest[0]] = {"side": PositionSide.LONG, "notional": target}

        return targets

    def check_funding_rate_change(self) -> List[str]:
        """检查费率是否有显著变化"""
        changed_symbols = []

        for symbol, current_rate in self.current_funding_rates.items():
            prev_rate = self.previous_funding_rates.get(symbol, current_rate)
            change = abs(current_rate - prev_rate)

            if change >= self.config.funding_rate_change_threshold:
                changed_symbols.append(symbol)
                logger.info(f"[费率变化] {symbol}: {prev_rate*100:+.4f}% -> {current_rate*100:+.4f}% (变化 {change*100:.4f}%)")

        return changed_symbols

    def get_portfolio_delta(self) -> Tuple[float, float, float]:
        """获取组合Delta"""
        total_long = 0.0
        total_short = 0.0

        for symbol, pos in self.current_positions.items():
            if pos["side"] == PositionSide.LONG:
                total_long += pos["notional"]
            elif pos["side"] == PositionSide.SHORT:
                total_short += pos["notional"]

        return total_long, total_short, total_long - total_short

    def check_rebalance_needed(self) -> Tuple[bool, str]:
        """检查是否需要调仓"""
        # 1. 检查是否有待执行的调仓任务
        if self.pending_rebalances:
            return True, f"继续执行 {len(self.pending_rebalances)} 个调仓任务"

        # 2. 计算最优目标
        self.target_positions = self.calculate_optimal_targets()

        # 3. 检查方向是否需要调整
        for symbol, target in self.target_positions.items():
            current = self.current_positions.get(symbol, {"side": None, "notional": 0})

            # 方向不同且当前有仓位
            if current["side"] and current["side"] != target["side"] and current["notional"] > self.config.min_trade_notional:
                return True, f"{symbol} 需要调整方向: {current['side'].value} -> {target['side'].value}"

            # 仓位大小偏差
            if target["notional"] > 0:
                diff_pct = abs(current["notional"] - target["notional"]) / target["notional"]
                if diff_pct > 0.10:  # 10% 偏差
                    return True, f"{symbol} 仓位偏差 {diff_pct:.1%}"

        # 4. 检查Delta偏移
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short

        if total_exposure > 0:
            imbalance_pct = abs(net_delta) / total_exposure
            if imbalance_pct >= self.config.emergency_delta_pct:
                return True, f"紧急Delta调仓: {imbalance_pct:.1%}"
            if imbalance_pct >= self.config.delta_threshold_pct:
                return True, f"Delta偏差: {imbalance_pct:.1%}"

        return False, "无需调仓"

    def create_rebalance_tasks(self) -> List[PendingRebalance]:
        """创建调仓任务（分批执行）"""
        tasks = []
        self.target_positions = self.calculate_optimal_targets()

        for symbol, target in self.target_positions.items():
            current = self.current_positions.get(symbol, {"side": None, "notional": 0})

            # 情况1: 需要反向（平仓+开新仓）
            if current["side"] and current["side"] != target["side"] and current["notional"] > self.config.min_trade_notional:
                # 先平仓
                tasks.append(PendingRebalance(
                    symbol=symbol,
                    from_side=current["side"],
                    to_side=target["side"],  # 表示要变成这个方向
                    total_notional=current["notional"] + target["notional"],  # 平仓+开仓的总量
                    remaining_notional=current["notional"] + target["notional"],
                    reason=f"反向调仓 {current['side'].value} -> {target['side'].value}"
                ))

            # 情况2: 方向正确，调整大小
            elif current["side"] == target["side"] or current["notional"] < self.config.min_trade_notional:
                diff = target["notional"] - current["notional"]
                if abs(diff) >= self.config.min_trade_notional:
                    tasks.append(PendingRebalance(
                        symbol=symbol,
                        from_side=current["side"],
                        to_side=target["side"],
                        total_notional=abs(diff),
                        remaining_notional=abs(diff),
                        reason=f"{'加仓' if diff > 0 else '减仓'} {target['side'].value} ${abs(diff):.0f}"
                    ))

        return tasks

    def get_next_batch_actions(self) -> List[dict]:
        """获取下一批要执行的操作"""
        actions = []

        for task in self.pending_rebalances[:]:  # 复制列表以便修改
            if task.remaining_notional < self.config.min_trade_notional:
                self.pending_rebalances.remove(task)
                continue

            # 计算本批调仓金额
            batch_size = min(
                task.remaining_notional,
                task.total_notional / self.config.rebalance_batch_count,
                self.config.target_notional_per_side * self.config.max_single_trade_pct
            )
            batch_size = max(batch_size, self.config.min_trade_notional)

            price = self.current_prices.get(task.symbol, 0)
            if price <= 0:
                continue

            current = self.current_positions.get(task.symbol, {"side": None, "notional": 0})

            # 确定操作
            if current["side"] and current["side"] != task.to_side and current["notional"] > self.config.min_trade_notional:
                # 需要先平仓
                close_amount = min(batch_size, current["notional"])
                side = "SELL" if current["side"] == PositionSide.LONG else "BUY"
                actions.append({
                    "symbol": task.symbol,
                    "side": side,
                    "quantity": close_amount / price,
                    "notional": close_amount,
                    "reason": f"平仓 {current['side'].value}",
                    "reduce_only": True,
                    "task": task
                })
            else:
                # 开仓或加仓
                side = "BUY" if task.to_side == PositionSide.LONG else "SELL"
                actions.append({
                    "symbol": task.symbol,
                    "side": side,
                    "quantity": batch_size / price,
                    "notional": batch_size,
                    "reason": task.reason,
                    "task": task
                })

        return actions

    def update_task_progress(self, task: PendingRebalance, executed_notional: float):
        """更新任务进度"""
        task.remaining_notional -= executed_notional
        task.batches_completed += 1

        if task.remaining_notional < self.config.min_trade_notional:
            if task in self.pending_rebalances:
                self.pending_rebalances.remove(task)
                logger.info(f"[任务完成] {task.symbol} {task.reason}")

    def get_status(self) -> dict:
        """获取状态"""
        total_long, total_short, net_delta = self.get_portfolio_delta()
        total_exposure = total_long + total_short
        imbalance = abs(net_delta) / total_exposure if total_exposure > 0 else 0

        return {
            "total_long": total_long,
            "total_short": total_short,
            "net_delta": net_delta,
            "imbalance_pct": imbalance,
            "is_neutral": imbalance < self.config.delta_threshold_pct,
            "pending_tasks": len(self.pending_rebalances),
            "positions": {
                s: {
                    "side": p["side"].value if p["side"] else None,
                    "notional": p["notional"]
                }
                for s, p in self.current_positions.items()
            },
            "targets": {
                s: {
                    "side": t["side"].value,
                    "notional": t["notional"]
                }
                for s, t in self.target_positions.items()
            },
            "funding_rates": self.current_funding_rates,
            # 自适应参数
            "volatility": self.avg_volatility,
            "volatility_by_symbol": self.volatility.copy(),
            "adaptive_leverage": self.current_leverage,
            "adaptive_target": self.current_target_notional,
            "max_target": self.config.max_notional_per_side
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
        logger.error(f"获取价格失败 {symbol}: {e}")
        return None


def fetch_binance_24h_stats(symbol: str) -> Optional[dict]:
    """
    从Binance获取24小时统计数据（包含最高价、最低价）

    Returns:
        {
            "high": float,
            "low": float,
            "price": float,
            "priceChangePercent": float
        }
    """
    binance_symbol = symbol.replace("-", "")
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"

    ssl_ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return {
                "high": float(data.get("highPrice", 0)),
                "low": float(data.get("lowPrice", 0)),
                "price": float(data.get("lastPrice", 0)),
                "priceChangePercent": float(data.get("priceChangePercent", 0))
            }
    except Exception as e:
        logger.error(f"获取24h统计失败 {symbol}: {e}")
        return None


def calculate_volatility_from_24h(symbol: str) -> float:
    """
    从24小时统计数据计算波动率

    波动率 = (最高价 - 最低价) / 平均价
    """
    stats = fetch_binance_24h_stats(symbol)
    if not stats:
        return 0.0

    high = stats["high"]
    low = stats["low"]
    avg = (high + low) / 2

    if avg <= 0:
        return 0.0

    return (high - low) / avg


def fetch_apex_funding_rates(symbols: List[str], testnet: bool = False) -> Dict[str, float]:
    """获取Apex资金费率"""
    endpoint = "https://testnet.omni.apex.exchange" if testnet else "https://omni.apex.exchange"
    rates = {}

    ssl_ctx = ssl.create_default_context()
    for symbol in symbols:
        ticker_symbol = symbol.replace("-", "")
        url = f"{endpoint}/api/v3/ticker?symbol={ticker_symbol}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                tickers = data.get("data", [])
                if tickers:
                    rate = tickers[0].get("fundingRate")
                    if rate is not None:
                        rates[symbol] = float(rate)
        except Exception as e:
            logger.error(f"获取费率失败 {symbol}: {e}")

    return rates


# ============ 交易执行器 ============

class AdaptiveTrader:
    """自适应交易执行器"""

    def __init__(self, strategy: AdaptiveHedgingStrategy, testnet: bool = True, dry_run: bool = True):
        self.strategy = strategy
        self.testnet = testnet
        self.dry_run = dry_run

        self.pending_orders: Dict[str, dict] = {}
        self.last_funding_check = 0
        self.running = False

        # 初始化API客户端
        self.sign_client = None
        if not dry_run and SDK_AVAILABLE:
            self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        api_key = os.getenv("APEX_API_KEY", "")
        secret_key = os.getenv("APEX_SECRET_KEY", "")
        passphrase = os.getenv("APEX_PASSPHRASE", "")
        omni_key = os.getenv("APEX_OMNIKEY", "")

        if not all([api_key, secret_key, passphrase, omni_key]):
            logger.error("API密钥未配置")
            return

        try:
            endpoint = "https://testnet.omni.apex.exchange" if self.testnet else "https://omni.apex.exchange"
            self.sign_client = HttpPrivate(
                endpoint,
                api_key=api_key,
                api_secret=secret_key,
                passphrase=passphrase,
                omni_key=omni_key
            )
            # 获取账户ID
            account_info = self.sign_client.get_account_v3()
            account_id = account_info.get("data", {}).get("id")
            if account_id:
                logger.info(f"OMNIKEY签名客户端初始化成功")
                logger.info(f"账户ID: {account_id}")
        except Exception as e:
            logger.error(f"初始化客户端失败: {e}")

    def update_prices(self):
        """更新所有价格"""
        for symbol in self.strategy.config.symbols:
            price = fetch_binance_price(symbol)
            if price:
                self.strategy.update_price(symbol, price)

    def update_funding_rates(self):
        """更新资金费率"""
        rates = fetch_apex_funding_rates(
            self.strategy.config.symbols,
            testnet=self.testnet
        )
        if rates:
            self.strategy.update_funding_rates(rates)
            return True
        return False

    def init_volatility_from_24h(self):
        """使用24小时数据初始化波动率（启动时）"""
        logger.info("[初始化] 获取24小时波动率数据...")

        volatilities = []
        for symbol in self.strategy.config.symbols:
            vol = calculate_volatility_from_24h(symbol)
            if vol > 0:
                self.strategy.volatility[symbol] = vol
                volatilities.append(vol)
                logger.info(f"  {symbol}: {vol*100:.2f}%")

        if volatilities:
            self.strategy.avg_volatility = sum(volatilities) / len(volatilities)
            logger.info(f"  平均波动率: {self.strategy.avg_volatility*100:.2f}%")

            # 立即计算自适应参数
            self.strategy.current_leverage = self.strategy.get_adaptive_leverage()
            self.strategy.current_target_notional = self.strategy.get_adaptive_target_notional()

            logger.info(f"[自适应参数] 杠杆: {self.strategy.current_leverage}x | "
                        f"目标仓位: ${self.strategy.current_target_notional:,.0f}")

    def sync_positions(self):
        """同步持仓"""
        if self.dry_run or not self.sign_client:
            return

        try:
            result = self.sign_client.get_positions_v3()
            positions = result.get("data", {}).get("positions", [])

            for symbol in self.strategy.config.symbols:
                found = False
                for pos in positions:
                    if pos.get("symbol") == symbol:
                        qty = float(pos.get("size", 0))
                        side_str = pos.get("side", "")
                        side = PositionSide.LONG if side_str == "LONG" else PositionSide.SHORT if side_str == "SHORT" else None
                        self.strategy.update_position(symbol, qty, side)
                        found = True
                        break
                if not found:
                    self.strategy.update_position(symbol, 0, None)
        except Exception as e:
            logger.error(f"同步持仓失败: {e}")

    def place_limit_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Optional[str]:
        """下限价单"""
        price = self.strategy.current_prices.get(symbol, 0)
        if price <= 0:
            return None

        # 计算挂单价格
        offset = self.strategy.config.limit_order_offset_pct
        if side == "BUY":
            limit_price = price * (1 - offset)
        else:
            limit_price = price * (1 + offset)

        if self.dry_run:
            logger.info(f"[模拟] {symbol} {side} {quantity:.6f} @ {limit_price:.2f}")
            return f"dry_run_{time.time()}"

        if not self.sign_client:
            return None

        try:
            result = self.sign_client.create_order_v3(
                symbol=symbol,
                side=side,
                type="LIMIT",
                size=str(quantity),
                price=str(limit_price),
                reduceOnly=reduce_only,
                timeInForce="GTC"
            )

            if result.get("data", {}).get("orderId"):
                order_id = result["data"]["orderId"]
                logger.info(f"[限价单] {symbol} {side} {quantity:.6f} @ {limit_price:.2f} -> {order_id}")
                return order_id
            else:
                logger.error(f"[下单失败] {symbol}: {result}")
                return None
        except Exception as e:
            logger.error(f"[下单异常] {symbol}: {e}")
            return None

    def check_order_status(self, order_id: str) -> Optional[str]:
        """检查订单状态"""
        if self.dry_run:
            return "FILLED"

        if not self.sign_client:
            return None

        try:
            result = self.sign_client.get_order_v3(id=order_id)
            return result.get("data", {}).get("status")
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if self.dry_run:
            return True

        if not self.sign_client:
            return False

        try:
            self.sign_client.cancel_order_v3(id=order_id)
            return True
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def execute_batch(self, actions: List[dict]):
        """执行一批操作"""
        for action in actions:
            symbol = action["symbol"]
            side = action["side"]
            quantity = action["quantity"]
            notional = action["notional"]
            reduce_only = action.get("reduce_only", False)
            task = action.get("task")

            order_id = self.place_limit_order(symbol, side, quantity, reduce_only)

            if order_id:
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "notional": notional,
                    "time": time.time(),
                    "task": task,
                    "retry_count": 0
                }

    def check_pending_orders(self):
        """检查挂单状态"""
        timeout = self.strategy.config.order_timeout_seconds
        max_retry = self.strategy.config.max_retry_count
        current_time = time.time()

        orders_to_remove = []

        for order_id, info in list(self.pending_orders.items()):
            elapsed = current_time - info["time"]

            status = self.check_order_status(order_id)

            if status == "FILLED":
                logger.info(f"[成交] {info['symbol']} {info['side']} ${info['notional']:.0f}")
                orders_to_remove.append(order_id)

                # 更新任务进度
                if info.get("task"):
                    self.strategy.update_task_progress(info["task"], info["notional"])

                self.strategy.total_volume += info["notional"]
                self.strategy.trades_executed += 1

            elif status == "CANCELED":
                orders_to_remove.append(order_id)

            elif elapsed > timeout:
                # 超时重挂
                retry_count = info.get("retry_count", 0)

                if retry_count >= max_retry:
                    # 改用市价单
                    logger.warning(f"[市价] {info['symbol']} 重试{retry_count}次失败")
                    self.cancel_order(order_id)
                    orders_to_remove.append(order_id)
                    # TODO: 实现市价单
                else:
                    # 取消并重挂
                    self.cancel_order(order_id)
                    orders_to_remove.append(order_id)

                    # 重新挂单
                    new_order_id = self.place_limit_order(
                        info["symbol"], info["side"], info["quantity"],
                        info.get("reduce_only", False)
                    )
                    if new_order_id:
                        self.pending_orders[new_order_id] = {
                            **info,
                            "time": time.time(),
                            "retry_count": retry_count + 1
                        }
                        logger.info(f"[重挂] {info['symbol']} 第{retry_count+1}次")

        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

    def run_once(self):
        """运行一次"""
        # 1. 更新价格
        self.update_prices()

        # 2. 同步持仓
        self.sync_positions()

        # 3. 检查挂单
        self.check_pending_orders()

        # 4. 更新自适应参数（波动率、杠杆、目标仓位）
        self.strategy.update_adaptive_params()

        # 5. 检查费率（每5分钟）
        current_time = time.time()
        if current_time - self.last_funding_check >= self.strategy.config.funding_check_interval_seconds:
            self.last_funding_check = current_time
            self.update_funding_rates()

            # 检查费率变化
            changed = self.strategy.check_funding_rate_change()
            if changed:
                logger.info(f"[费率变化] 重新计算目标配置")

        # 6. 检查是否需要调仓
        need_rebalance, reason = self.strategy.check_rebalance_needed()

        if need_rebalance:
            logger.info(f"[调仓] {reason}")

            # 创建调仓任务（如果没有）
            if not self.strategy.pending_rebalances:
                tasks = self.strategy.create_rebalance_tasks()
                self.strategy.pending_rebalances.extend(tasks)
                for task in tasks:
                    logger.info(f"  -> {task.symbol}: {task.reason} (${task.total_notional:.0f})")

            # 执行下一批
            if not self.pending_orders:  # 等待上一批完成
                actions = self.strategy.get_next_batch_actions()
                if actions:
                    self.execute_batch(actions)

        # 7. 打印状态
        status = self.strategy.get_status()
        vol_str = f"波动率: {status['volatility']*100:.2f}%" if status['volatility'] > 0 else "波动率: 计算中"
        logger.info(f"[状态] 多头: ${status['total_long']:,.0f} | 空头: ${status['total_short']:,.0f} | "
                    f"Delta: ${status['net_delta']:,.0f} ({status['imbalance_pct']:.1%}) | "
                    f"{'中性' if status['is_neutral'] else '失衡'} | "
                    f"{vol_str} | {status['adaptive_leverage']}x | "
                    f"目标: ${status['adaptive_target']:,.0f}/{status['max_target']:,.0f}")

    def run(self):
        """主循环"""
        self.running = True
        logger.info("=" * 60)
        logger.info("自适应Delta中性对冲策略启动")
        logger.info("=" * 60)
        logger.info(f"模式: {'模拟' if self.dry_run else '实盘'}")
        logger.info(f"网络: {'测试网' if self.testnet else '主网'}")
        logger.info(f"最大仓位: 单边 ${self.strategy.config.max_notional_per_side:,.0f} (自适应调整)")
        logger.info(f"杠杆范围: {self.strategy.config.min_leverage}x - {self.strategy.config.max_leverage}x (自适应)")
        logger.info(f"币种: {', '.join(self.strategy.config.symbols)}")
        logger.info(f"方向: 根据资金费率自动选择")
        logger.info("=" * 60)

        # 初始化
        self.update_prices()
        self.update_funding_rates()

        # 初始化波动率（使用24小时数据）
        self.init_volatility_from_24h()

        # 显示当前费率
        logger.info("[当前资金费率]")
        for symbol, rate in sorted(self.strategy.current_funding_rates.items(), key=lambda x: x[1], reverse=True):
            direction = "SHORT有利" if rate > 0 else "LONG有利" if rate < 0 else "中性"
            logger.info(f"  {symbol}: {rate*100:+.6f}% ({direction})")

        # 显示目标配置
        targets = self.strategy.calculate_optimal_targets()
        logger.info("[目标配置]")
        for symbol, target in targets.items():
            if target["notional"] > 0:
                logger.info(f"  {symbol}: {target['side'].value} ${target['notional']:,.0f}")

        try:
            while self.running:
                self.run_once()
                time.sleep(self.strategy.config.check_interval_seconds)
        except KeyboardInterrupt:
            logger.info("收到停止信号")
            self.running = False

        logger.info(f"[停止] 总交易量: ${self.strategy.total_volume:,.0f} | 交易次数: {self.strategy.trades_executed}")


def main():
    parser = argparse.ArgumentParser(description="自适应Delta中性对冲策略")
    parser.add_argument('--dry-run', action='store_true', help='模拟模式')
    parser.add_argument('--max-target', type=float, default=50000, help='单边最大仓位 (默认50000)')
    parser.add_argument('--min-target', type=float, default=10000, help='单边最小仓位 (默认10000)')
    parser.add_argument('--min-leverage', type=int, default=20, help='最小杠杆 (默认20)')
    parser.add_argument('--max-leverage', type=int, default=50, help='最大杠杆 (默认50)')
    parser.add_argument('--interval', type=int, default=30, help='检查间隔秒数 (默认30)')
    args = parser.parse_args()

    testnet = os.getenv("APEX_TESTNET", "true").lower() == "true"

    config = AdaptiveConfig(
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        max_notional_per_side=args.max_target,
        min_notional_per_side=args.min_target,
        min_leverage=args.min_leverage,
        max_leverage=args.max_leverage,
        check_interval_seconds=args.interval
    )

    strategy = AdaptiveHedgingStrategy(config)
    trader = AdaptiveTrader(strategy, testnet=testnet, dry_run=args.dry_run)

    if not testnet and not args.dry_run:
        print("\n[!] 警告: 主网实盘交易模式!")
        confirm = input("确认? (输入 YES): ")
        if confirm != "YES":
            print("已取消")
            return

    trader.run()


if __name__ == "__main__":
    main()
