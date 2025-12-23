"""
中性对冲交易执行器 - Neutral Hedging Trader
============================================

执行跨币种中性对冲策略
- BTC多头 + ETH空头 + SOL多头
- 自动调仓保持Delta中性
- 产生交易量用于刷KYC/VIP等级

使用Python 3.12运行:
  py -3.12 hedging_trader.py
"""
import os
import sys
import time
import json
import ssl
import urllib.request
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from neutral_hedging_strategy import (
    NeutralHedgingStrategy, HedgingConfig, AssetConfig,
    PositionSide, RebalanceAction
)

# 设置日志
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, 'hedging.log')


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        FlushFileHandler(LOG_FILE, encoding='utf-8'),
        FlushStreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_env():
    """加载环境变量"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


load_env()


def fetch_binance_price(symbol: str) -> Optional[float]:
    """从Binance获取最新价格"""
    binance_symbol = symbol.replace("-", "")
    if binance_symbol.startswith("1000"):
        binance_symbol = binance_symbol[4:]

    ssl_ctx = ssl.create_default_context()
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return float(data.get("price", 0))
    except Exception as e:
        logger.error(f"获取Binance价格失败 {symbol}: {e}")
        return None


def fetch_apex_funding_info(symbol: str, testnet: bool = False) -> Optional[dict]:
    """从Apex获取完整的资金费率信息 (包括结算时间)

    Args:
        symbol: 交易对 (如 BTC-USDT)
        testnet: 是否使用测试网

    Returns:
        {
            "rate": float,           # 当前资金费率
            "predicted_rate": float, # 预测费率
            "next_funding_time": datetime,  # 下次结算时间(UTC)
            "seconds_until_funding": int    # 距离结算的秒数
        }
        失败返回None
    """
    if testnet:
        endpoint = "https://testnet.omni.apex.exchange"
    else:
        endpoint = "https://omni.apex.exchange"

    # 转换 symbol 格式: BTC-USDT -> BTCUSDT
    ticker_symbol = symbol.replace("-", "")

    ssl_ctx = ssl.create_default_context()
    url = f"{endpoint}/api/v3/ticker?symbol={ticker_symbol}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
            tickers = data.get("data", [])
            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                rate = ticker.get("fundingRate")
                predicted = ticker.get("predictedFundingRate")
                next_time_str = ticker.get("nextFundingTime", "")

                if rate is None:
                    return None

                result = {
                    "rate": float(rate),
                    "predicted_rate": float(predicted) if predicted else 0.0,
                    "next_funding_time": None,
                    "seconds_until_funding": 0
                }

                # 解析下次结算时间
                if next_time_str:
                    try:
                        # 格式: "2025-12-23T08:00:00.000Z"
                        next_time = datetime.fromisoformat(next_time_str.replace("Z", "+00:00"))
                        result["next_funding_time"] = next_time
                        # 计算距离结算的秒数
                        now = datetime.now(next_time.tzinfo)
                        delta = next_time - now
                        result["seconds_until_funding"] = max(0, int(delta.total_seconds()))
                    except Exception:
                        pass

                return result
    except Exception as e:
        logger.error(f"获取Apex资金费率失败 {symbol}: {e}")
    return None


def fetch_apex_funding_rate(symbol: str, testnet: bool = False) -> Optional[float]:
    """从Apex获取当前资金费率 (实时费率，非历史)

    Args:
        symbol: 交易对 (如 BTC-USDT)
        testnet: 是否使用测试网

    Returns:
        资金费率 (如 0.0001 = 0.01%)，失败返回None
    """
    info = fetch_apex_funding_info(symbol, testnet)
    if info:
        return info["rate"]
    return None


def fetch_all_funding_rates(symbols: List[str], testnet: bool = False) -> Dict[str, float]:
    """获取多个币种的资金费率

    Returns:
        {symbol: rate} 字典
    """
    rates = {}
    for symbol in symbols:
        rate = fetch_apex_funding_rate(symbol, testnet)
        if rate is not None:
            rates[symbol] = rate
    return rates


def fetch_all_funding_info(symbols: List[str], testnet: bool = False) -> Dict[str, dict]:
    """获取多个币种的完整资金费率信息（包括结算时间）

    Returns:
        {symbol: {rate, predicted_rate, next_funding_time, seconds_until_funding}} 字典
    """
    info_dict = {}
    for symbol in symbols:
        info = fetch_apex_funding_info(symbol, testnet)
        if info is not None:
            info_dict[symbol] = info
    return info_dict


def calculate_optimal_positions(funding_rates: Dict[str, float],
                                total_notional: float = 400.0) -> List[AssetConfig]:
    """根据资金费率计算最优持仓配置

    策略:
    - 正费率币种做空 (空头收取费率)
    - 负费率币种做多 (多头收取费率)
    - 保持Delta中性 (多头总额 = 空头总额)

    Args:
        funding_rates: {symbol: rate} 资金费率字典
        total_notional: 总名义价值 (多空各一半)

    Returns:
        AssetConfig列表
    """
    if not funding_rates:
        return []

    # 按费率排序 (高到低)
    sorted_rates = sorted(funding_rates.items(), key=lambda x: x[1], reverse=True)

    # 分配策略
    # - 费率最高的做空 (赚取费率)
    # - 费率最低的做多 (成本最低)
    # - 中间的根据正负决定

    assets = []
    half_notional = total_notional / 2  # 多空各一半

    if len(sorted_rates) == 3:
        highest = sorted_rates[0]  # 费率最高
        middle = sorted_rates[1]   # 中间
        lowest = sorted_rates[2]   # 费率最低

        # 判断中间币的方向
        if middle[1] > 0:
            # 中间币也是正费率，做空
            # 最低费率币做多 $200，其他两个做空各 $100
            assets = [
                AssetConfig(lowest[0], PositionSide.LONG, 200.0, weight=1.0),
                AssetConfig(highest[0], PositionSide.SHORT, 100.0, weight=1.0),
                AssetConfig(middle[0], PositionSide.SHORT, 100.0, weight=1.0),
            ]
        else:
            # 中间币负费率，做多
            # 最高费率币做空 $200，其他两个做多各 $100
            assets = [
                AssetConfig(lowest[0], PositionSide.LONG, 100.0, weight=1.0),
                AssetConfig(middle[0], PositionSide.LONG, 100.0, weight=1.0),
                AssetConfig(highest[0], PositionSide.SHORT, 200.0, weight=1.0),
            ]
    elif len(sorted_rates) == 2:
        # 两个币种：高费率做空，低费率做多
        highest = sorted_rates[0]
        lowest = sorted_rates[1]
        assets = [
            AssetConfig(lowest[0], PositionSide.LONG, half_notional, weight=1.0),
            AssetConfig(highest[0], PositionSide.SHORT, half_notional, weight=1.0),
        ]
    else:
        # 单个币种：无法做中性对冲
        logger.warning("需要至少2个币种才能进行中性对冲")

    return assets


# 导入官方SDK
try:
    from apexomni.http_public import HttpPublic
    from apexomni.http_private import HttpPrivate
    from apexomni.http_private_sign import HttpPrivateSign
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("apexomni SDK not found. Install with: pip install apexomni")


class ApexTrader:
    """Apex Exchange交易接口"""

    NETWORK_ID_MAINNET = 42161
    NETWORK_ID_TESTNET = 421614

    def __init__(self, api_key: str, secret_key: str, passphrase: str,
                 omnikey: str = "", testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.omnikey = omnikey
        self.testnet = testnet

        if testnet:
            self.endpoint = "https://testnet.omni.apex.exchange"
            self.network_id = self.NETWORK_ID_TESTNET
        else:
            self.endpoint = "https://omni.apex.exchange"
            self.network_id = self.NETWORK_ID_MAINNET

        self.public_client = HttpPublic(self.endpoint)
        self.private_client = HttpPrivate(
            self.endpoint,
            network_id=self.network_id,
            api_key_credentials={
                'key': api_key,
                'secret': secret_key,
                'passphrase': passphrase
            }
        )

        self.sign_client = None
        if omnikey:
            try:
                self.sign_client = HttpPrivateSign(
                    self.endpoint,
                    network_id=self.network_id,
                    zk_seeds=omnikey,
                    zk_l2Key=omnikey,
                    api_key_credentials={
                        'key': api_key,
                        'secret': secret_key,
                        'passphrase': passphrase
                    }
                )
                self.sign_client.configs_v3()
                logger.info("OMNIKEY签名客户端初始化成功")
            except Exception as e:
                logger.error(f"OMNIKEY签名客户端初始化失败: {e}")
                self.sign_client = None

    def get_account(self) -> dict:
        """获取账户信息"""
        try:
            if self.sign_client:
                result = self.sign_client.get_account_v3()
            else:
                result = self.private_client.get_account_v2()
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"获取账户失败: {e}")
            return {"success": False, "error": str(e)}

    def get_positions(self) -> List[dict]:
        """获取持仓"""
        result = self.get_account()
        if result.get("success"):
            data = result.get("data", {})
            positions = data.get("positions", [])
            if positions:
                return [p for p in positions if float(p.get("size", 0)) != 0]
        return []

    def _round_size_to_step(self, size: float, symbol: str) -> float:
        """将数量圆整到stepSize的倍数"""
        try:
            if self.sign_client and hasattr(self.sign_client, 'configV3') and self.sign_client.configV3:
                config = self.sign_client.configV3
                contracts = config.get('contractConfig', {}).get('perpetualContract', [])
                for c in contracts:
                    if c.get('symbol') == symbol or c.get('symbolDisplayName') == symbol:
                        step_size = float(c.get('stepSize', 0.001))
                        rounded = int(size / step_size) * step_size
                        decimals = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
                        return round(rounded, decimals)
        except Exception as e:
            logger.warning(f"获取stepSize失败: {e}")
        return size

    def _round_price_to_tick(self, price: float, symbol: str) -> float:
        """将价格圆整到tickSize的倍数"""
        try:
            if self.sign_client and hasattr(self.sign_client, 'configV3') and self.sign_client.configV3:
                config = self.sign_client.configV3
                contracts = config.get('contractConfig', {}).get('perpetualContract', [])
                for c in contracts:
                    if c.get('symbol') == symbol or c.get('symbolDisplayName') == symbol:
                        tick_size = float(c.get('tickSize', 0.01))
                        rounded = round(price / tick_size) * tick_size
                        decimals = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
                        return round(rounded, decimals)
        except Exception as e:
            logger.warning(f"获取tickSize失败: {e}")
        return price

    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "MARKET", price: float = None,
                    reduce_only: bool = False) -> dict:
        """下单

        Args:
            symbol: 交易对
            side: BUY/SELL
            size: 数量
            order_type: MARKET/LIMIT
            price: 限价单价格 (LIMIT单必须)
            reduce_only: 是否只减仓
        """
        if not self.sign_client:
            logger.error("OMNIKEY未配置，无法下单")
            return {"success": False, "error": "OMNIKEY未配置"}

        try:
            size = self._round_size_to_step(size, symbol)

            order_params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "size": str(size),
                "reduceOnly": reduce_only,
                "timestampSeconds": time.time(),
            }

            # 限价单需要指定价格
            if order_type.upper() == "LIMIT":
                if price is None:
                    logger.error("限价单必须指定价格")
                    return {"success": False, "error": "限价单必须指定价格"}
                price = self._round_price_to_tick(price, symbol)
                order_params["price"] = str(price)
                logger.info(f"[限价单] {symbol} {side} {size} @ {price}")
            else:
                # 市价单也需要price参数（SDK要求）
                current_price = fetch_binance_price(symbol)
                if current_price:
                    current_price = self._round_price_to_tick(current_price, symbol)
                    order_params["price"] = str(current_price)

            result = self.sign_client.create_order_v3(**order_params)
            order_id = result.get('data', {}).get('id', 'unknown')
            logger.info(f"下单成功: {order_id}")
            return {"success": True, "data": result.get("data", {}), "order_id": order_id}

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {"success": False, "error": str(e)}

    def get_open_orders(self, symbol: str = None) -> List[dict]:
        """获取未成交订单"""
        try:
            if self.sign_client:
                result = self.sign_client.open_orders_v3()
                orders = result.get("data", {}).get("orders", [])
                if symbol:
                    orders = [o for o in orders if o.get("symbol") == symbol]
                return orders
            return []
        except Exception as e:
            logger.error(f"获取订单失败: {e}")
            return []

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """查询订单状态"""
        try:
            if self.sign_client:
                result = self.sign_client.get_order_v3(id=order_id)
                return result.get("data", {})
            return None
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            if self.sign_client:
                result = self.sign_client.delete_order_v3(id=order_id)
                logger.info(f"取消订单成功: {order_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def cancel_all_orders(self, symbol: str = None) -> int:
        """取消所有订单"""
        try:
            if self.sign_client:
                if symbol:
                    result = self.sign_client.delete_open_orders_v3(symbol=symbol)
                else:
                    result = self.sign_client.delete_open_orders_v3()
                cancelled = result.get("data", {}).get("cancelledCount", 0)
                logger.info(f"取消了 {cancelled} 个订单")
                return cancelled
            return 0
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return 0


class HedgingTrader:
    """中性对冲交易器"""

    def __init__(self, strategy: NeutralHedgingStrategy, trader: ApexTrader,
                 dry_run: bool = True, check_interval: int = 60,
                 use_limit_orders: bool = False, price_offset_pct: float = 0.0005,
                 dynamic_funding: bool = False, funding_check_interval: int = 3600,
                 testnet: bool = True, pre_settlement_check_minutes: int = 5):
        """
        Args:
            strategy: 对冲策略实例
            trader: 交易接口
            dry_run: 模拟模式
            check_interval: 检查间隔(秒)
            use_limit_orders: 是否使用限价单 (Maker)
            price_offset_pct: 限价单价格偏移百分比 (默认0.05% = 5 bps)
                - BUY: 挂在 current_price * (1 - offset) 处
                - SELL: 挂在 current_price * (1 + offset) 处
            dynamic_funding: 是否根据资金费率动态调整持仓方向
            funding_check_interval: 资金费率检查间隔(秒)，默认1小时
            testnet: 是否使用测试网
            pre_settlement_check_minutes: 结算前多少分钟触发检查 (默认5分钟)
        """
        self.strategy = strategy
        self.trader = trader
        self.dry_run = dry_run
        self.check_interval = check_interval
        self.use_limit_orders = use_limit_orders
        self.price_offset_pct = price_offset_pct
        self.dynamic_funding = dynamic_funding
        self.funding_check_interval = funding_check_interval
        self.testnet = testnet
        self.running = False
        self.pre_settlement_check_minutes = pre_settlement_check_minutes

        # 挂单跟踪
        self.pending_orders: Dict[str, dict] = {}  # order_id -> order_info

        # 资金费率跟踪
        self.last_funding_check = 0
        self.current_funding_rates: Dict[str, float] = {}
        self.current_funding_info: Dict[str, dict] = {}  # 包含结算时间的完整信息
        self.last_settlement_hour_checked = -1  # 防止同一小时多次触发

        # 统计
        self.trades_executed = 0
        self.total_volume = 0.0
        self.start_time = None

    def sync_positions(self):
        """同步交易所持仓到策略"""
        positions = self.trader.get_positions()

        for symbol in self.strategy.positions.keys():
            # 查找对应持仓
            found = False
            for p in positions:
                if p.get("symbol") == symbol:
                    size = float(p.get("size", 0))
                    entry_price = float(p.get("entryPrice", 0))
                    side = p.get("side", "")

                    # 判断是多头还是空头
                    expected_side = self.strategy.positions[symbol].side
                    actual_side = PositionSide.LONG if side == "LONG" else PositionSide.SHORT

                    if actual_side == expected_side:
                        self.strategy.update_position(symbol, abs(size), entry_price)
                        logger.info(f"[{symbol}] 同步持仓: {actual_side.value} {abs(size):.6f} @ {entry_price:.2f}")
                    else:
                        logger.warning(f"[{symbol}] 持仓方向不匹配! 期望{expected_side.value}, 实际{actual_side.value}")

                    found = True
                    break

            if not found:
                # 无持仓
                self.strategy.update_position(symbol, 0, 0)

    def check_settlement_time(self) -> tuple:
        """检查是否接近结算时间

        Returns:
            (is_near_settlement, seconds_until, next_time_str)
            - is_near_settlement: 是否接近结算时间
            - seconds_until: 距离下次结算的秒数
            - next_time_str: 下次结算时间字符串
        """
        symbols = [asset.symbol for asset in self.strategy.config.assets]

        # 获取任意一个币种的结算时间信息（所有币种结算时间相同）
        info = fetch_apex_funding_info(symbols[0], testnet=self.testnet)

        if not info or info["next_funding_time"] is None:
            return False, 0, ""

        seconds_until = info["seconds_until_funding"]
        pre_settlement_seconds = self.pre_settlement_check_minutes * 60

        # 格式化时间显示
        next_time = info["next_funding_time"]
        next_time_str = next_time.strftime("%H:%M:%S UTC")

        # 检查是否在结算前X分钟内
        is_near = 0 < seconds_until <= pre_settlement_seconds

        return is_near, seconds_until, next_time_str

    def check_and_update_funding_rates(self, force: bool = False) -> bool:
        """检查资金费率并根据需要调整策略

        Args:
            force: 强制检查（忽略时间间隔限制）

        Returns:
            True if strategy needs to be updated, False otherwise
        """
        if not self.dynamic_funding:
            return False

        current_time = time.time()

        # 检查是否接近结算时间
        is_near_settlement, seconds_until, next_time_str = self.check_settlement_time()

        # 结算时间感知：接近结算时强制检查
        if is_near_settlement:
            # 防止同一结算周期内多次触发
            current_hour = datetime.now().hour
            if self.last_settlement_hour_checked == current_hour:
                logger.debug(f"[结算感知] 本小时已检查过，跳过")
            else:
                logger.info(f"[结算感知] 距离结算还有 {seconds_until}秒 ({seconds_until//60}分{seconds_until%60}秒)")
                logger.info(f"[结算感知] 下次结算时间: {next_time_str}")
                logger.info(f"[结算感知] 在结算前自动检查资金费率!")
                self.last_settlement_hour_checked = current_hour
                force = True

        # 检查是否需要更新费率（普通间隔检查或强制检查）
        if not force and current_time - self.last_funding_check < self.funding_check_interval:
            return False

        self.last_funding_check = current_time

        # 获取所有币种的完整资金费率信息
        symbols = [asset.symbol for asset in self.strategy.config.assets]
        all_info = fetch_all_funding_info(symbols, testnet=self.testnet)

        if not all_info:
            logger.warning("[费率检查] 获取资金费率失败")
            return False

        # 记录完整信息
        self.current_funding_info = all_info
        self.current_funding_rates = {s: info["rate"] for s, info in all_info.items()}

        logger.info("[费率检查] 当前资金费率:")
        for symbol, info in sorted(all_info.items(), key=lambda x: x[1]["rate"], reverse=True):
            rate = info["rate"]
            predicted = info["predicted_rate"]
            direction = "SHORT有利" if rate > 0 else "LONG有利" if rate < 0 else "中性"
            logger.info(f"  {symbol}: {rate*100:+.6f}% ({direction}) | 预测: {predicted*100:+.6f}%")

        # 显示下次结算时间
        if all_info:
            first_info = list(all_info.values())[0]
            if first_info["next_funding_time"]:
                next_time = first_info["next_funding_time"]
                secs = first_info["seconds_until_funding"]
                logger.info(f"  下次结算: {next_time.strftime('%H:%M:%S UTC')} (还有 {secs//60}分{secs%60}秒)")

        # 计算最优配置
        optimal_assets = calculate_optimal_positions(self.current_funding_rates, total_notional=8000.0)

        if not optimal_assets:
            return False

        # 检查是否需要调整
        needs_update = False
        for optimal in optimal_assets:
            current = self.strategy.positions.get(optimal.symbol)
            if current and current.side != optimal.side:
                logger.info(f"[费率调整] {optimal.symbol}: {current.side.value} -> {optimal.side.value}")
                needs_update = True

        if needs_update:
            logger.info("[费率调整] 检测到需要调整持仓方向!")
            # 打印新配置
            logger.info("[费率调整] 新配置:")
            for asset in optimal_assets:
                logger.info(f"  {asset.symbol}: {asset.side.value} ${asset.target_notional:.0f}")

        return needs_update

    def execute_funding_rate_adjustment(self) -> List[RebalanceAction]:
        """执行资金费率调整，生成平仓和开仓操作

        当需要反向持仓时:
        1. 先平掉当前仓位
        2. 再开反向仓位

        Returns:
            需要执行的操作列表
        """
        if not self.current_funding_rates:
            return []

        actions = []
        optimal_assets = calculate_optimal_positions(self.current_funding_rates, total_notional=8000.0)

        for optimal in optimal_assets:
            current = self.strategy.positions.get(optimal.symbol)
            if not current:
                continue

            current_price = self.strategy.prices.get(optimal.symbol, 0)
            if current_price <= 0:
                current_price = fetch_binance_price(optimal.symbol)
                if current_price:
                    self.strategy.update_price(optimal.symbol, current_price)

            if current_price <= 0:
                logger.error(f"[费率调整] 无法获取 {optimal.symbol} 价格")
                continue

            # 检查是否需要反向
            if current.side != optimal.side:
                current_notional = current.quantity * current_price

                # 1. 平掉当前仓位
                if current.quantity > 0:
                    close_side = "SELL" if current.side == PositionSide.LONG else "BUY"
                    actions.append(RebalanceAction(
                        symbol=optimal.symbol,
                        side=close_side,
                        quantity=current.quantity,
                        notional=current_notional,
                        reason=f"Close {current.side.value} for direction change"
                    ))

                # 2. 开反向仓位
                new_quantity = optimal.target_notional / current_price
                open_side = "BUY" if optimal.side == PositionSide.LONG else "SELL"
                actions.append(RebalanceAction(
                    symbol=optimal.symbol,
                    side=open_side,
                    quantity=new_quantity,
                    notional=optimal.target_notional,
                    reason=f"Open {optimal.side.value} based on funding rate"
                ))

                # 更新策略配置
                for i, asset in enumerate(self.strategy.config.assets):
                    if asset.symbol == optimal.symbol:
                        self.strategy.config.assets[i] = optimal
                        self.strategy.positions[optimal.symbol].side = optimal.side
                        self.strategy.positions[optimal.symbol].target_notional = optimal.target_notional
                        break

        return actions

    def check_pending_orders(self, max_wait_seconds: int = 15):
        """检查挂单状态，超时未成交则取消并以更接近市价的价格重挂

        Args:
            max_wait_seconds: 最大等待时间，超过则取消订单并重挂（默认15秒）
        """
        if not self.pending_orders or self.dry_run:
            return

        current_time = time.time()
        orders_to_remove = []
        orders_to_retry = []

        for order_id, order_info in list(self.pending_orders.items()):
            elapsed = current_time - order_info["time"]

            # 查询订单状态
            status = self.trader.get_order_status(order_id)
            if status:
                order_status = status.get("status", "")

                if order_status == "FILLED":
                    # 已成交
                    logger.info(f"[订单成交] {order_info['symbol']} {order_info['side']} "
                               f"@ {order_info['price']:.2f}")
                    orders_to_remove.append(order_id)

                elif order_status == "CANCELED":
                    # 已取消
                    logger.info(f"[订单取消] {order_id}")
                    orders_to_remove.append(order_id)

                elif order_status in ["PENDING", "OPEN"]:
                    # 仍在挂单中
                    if elapsed > max_wait_seconds:
                        # 超时，需要取消并重挂
                        retry_count = order_info.get("retry_count", 0)
                        logger.warning(f"[订单超时] {order_info['symbol']} 等待 {elapsed:.0f}s，准备重挂 (第{retry_count+1}次)")
                        orders_to_retry.append((order_id, order_info))
                    else:
                        remaining = max_wait_seconds - elapsed
                        logger.debug(f"[挂单中] {order_info['symbol']} 剩余 {remaining:.0f}s")

        # 移除已完成的订单
        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

        # 处理超时订单：取消并重新挂单（价格更接近市价）
        for order_id, order_info in orders_to_retry:
            # 1. 取消原订单
            if self.trader.cancel_order(order_id):
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]

                retry_count = order_info.get("retry_count", 0) + 1

                # 重试超过3次，直接用市价单确保成交
                if retry_count > 3:
                    logger.warning(f"[市价成交] {order_info['symbol']} 重试{retry_count}次失败，改用市价单")
                    result = self.trader.place_order(
                        symbol=order_info["symbol"],
                        side=order_info["side"],
                        size=order_info["quantity"],
                        order_type="MARKET"
                    )
                    if result.get("success"):
                        logger.info(f"[市价成交] {order_info['symbol']} {order_info['side']} 成功")
                    else:
                        logger.error(f"[市价失败] {order_info['symbol']}: {result.get('error')}")
                    continue

                # 2. 获取最新价格
                current_price = fetch_binance_price(order_info["symbol"])
                if current_price and current_price > 0:
                    # 3. 计算新的挂单价格（每次重试更接近市价）
                    # 第1次: 0.03%, 第2次: 0.01%, 第3次: 0.005%
                    if retry_count == 1:
                        adjusted_offset = 0.0003  # 0.03%
                    elif retry_count == 2:
                        adjusted_offset = 0.0001  # 0.01%
                    else:
                        adjusted_offset = 0.00005  # 0.005%

                    if order_info["side"] == "BUY":
                        new_price = current_price * (1 - adjusted_offset)
                    else:
                        new_price = current_price * (1 + adjusted_offset)

                    logger.info(f"[重挂] {order_info['symbol']} 偏移: {adjusted_offset*100:.3f}% (第{retry_count}次)")

                    # 4. 重新挂单
                    result = self.trader.place_order(
                        symbol=order_info["symbol"],
                        side=order_info["side"],
                        size=order_info["quantity"],
                        order_type="LIMIT",
                        price=new_price
                    )

                    if result.get("success"):
                        new_order_id = result.get("order_id")
                        logger.info(f"[重新挂单] {order_info['symbol']} {order_info['side']} @ {new_price:.2f}")
                        self.pending_orders[new_order_id] = {
                            "symbol": order_info["symbol"],
                            "side": order_info["side"],
                            "quantity": order_info["quantity"],
                            "price": new_price,
                            "notional": order_info["notional"],
                            "time": time.time(),
                            "retry_count": retry_count
                        }
                    else:
                        logger.error(f"[重挂失败] {order_info['symbol']}: {result.get('error')}")

    def update_prices(self):
        """更新所有币种价格"""
        for symbol in self.strategy.positions.keys():
            price = fetch_binance_price(symbol)
            if price:
                self.strategy.update_price(symbol, price)
                logger.debug(f"[{symbol}] 价格: {price:.2f}")
            else:
                logger.warning(f"[{symbol}] 获取价格失败")

    def execute_action(self, action: RebalanceAction) -> bool:
        """执行调仓操作"""
        if self.dry_run:
            logger.info(f"[模拟] {action}")
            return True

        # 获取当前价格
        current_price = self.strategy.prices.get(action.symbol, 0)
        if current_price <= 0:
            current_price = fetch_binance_price(action.symbol)

        if self.use_limit_orders and current_price > 0:
            # 限价单模式 - 挂单做 Maker
            # 为了快速成交：买单挂高一点，卖单挂低一点（接近市价但仍是maker）
            if action.side == "BUY":
                # 买单挂在略低于市价的位置（容易被卖方吃掉）
                limit_price = current_price * (1 - self.price_offset_pct)
            else:
                # 卖单挂在略高于市价的位置（容易被买方吃掉）
                limit_price = current_price * (1 + self.price_offset_pct)

            result = self.trader.place_order(
                symbol=action.symbol,
                side=action.side,
                size=action.quantity,
                order_type="LIMIT",
                price=limit_price
            )

            if result.get("success"):
                order_id = result.get("order_id")
                logger.info(f"[限价单] 挂单成功: {action.symbol} {action.side} @ {limit_price:.2f}")
                # 记录挂单
                self.pending_orders[order_id] = {
                    "symbol": action.symbol,
                    "side": action.side,
                    "quantity": action.quantity,
                    "price": limit_price,
                    "notional": action.notional,
                    "time": time.time()
                }
                self.trades_executed += 1
                self.total_volume += action.notional
                return True
            else:
                logger.error(f"[限价单] 挂单失败: {action}, 错误: {result.get('error')}")
                return False
        else:
            # 市价单模式
            result = self.trader.place_order(
                symbol=action.symbol,
                side=action.side,
                size=action.quantity,
                order_type="MARKET"
            )

            if result.get("success"):
                logger.info(f"[市价单] 执行成功: {action}")
                self.trades_executed += 1
                self.total_volume += action.notional
                return True
            else:
                logger.error(f"[市价单] 执行失败: {action}, 错误: {result.get('error')}")
                return False

    def run_once(self):
        """运行一次检查和调仓"""
        # 0. 检查挂单状态（限价单模式）
        if self.use_limit_orders and not self.dry_run:
            self.check_pending_orders(max_wait_seconds=15)

        # 1. 更新价格
        self.update_prices()

        # 2. 同步持仓（非模拟模式）
        if not self.dry_run:
            self.sync_positions()

        # 3. 检查资金费率并调整策略（动态费率模式）
        if self.dynamic_funding:
            needs_adjustment = self.check_and_update_funding_rates()
            if needs_adjustment:
                # 执行费率调整操作
                adjustment_actions = self.execute_funding_rate_adjustment()
                if adjustment_actions:
                    logger.info(f"[费率调整] 需要执行 {len(adjustment_actions)} 个操作:")
                    for action in adjustment_actions:
                        logger.info(f"  {action}")
                        success = self.execute_action(action)
                        if success and self.dry_run:
                            # 模拟模式：手动更新持仓
                            pos = self.strategy.positions[action.symbol]
                            price = self.strategy.prices.get(action.symbol, 0)
                            if price > 0:
                                # 平仓操作后数量为0，开仓操作后为新数量
                                if "Close" in action.reason:
                                    self.strategy.update_position(action.symbol, 0, price)
                                else:
                                    self.strategy.update_position(action.symbol, action.quantity, price)
                    self.strategy.record_rebalance(adjustment_actions)
                    return  # 费率调整后跳过常规调仓

        # 4. 打印当前状态
        summary = self.strategy.get_portfolio_summary()
        logger.info(f"[状态] 多头: ${summary['total_long']:.0f} | "
                   f"空头: ${summary['total_short']:.0f} | "
                   f"净Delta: ${summary['net_delta']:.0f} | "
                   f"偏差: {summary['imbalance_pct']:.1%} | "
                   f"中性: {'是' if summary['is_neutral'] else '否'}")

        # 5. 检查是否需要调仓
        should, reason = self.strategy.should_rebalance()
        logger.info(f"[调仓检查] {reason}")

        if should:
            # 6. 计算调仓操作
            actions = self.strategy.calculate_rebalance_actions()

            if actions:
                logger.info(f"[调仓] 需要执行 {len(actions)} 个操作:")
                for action in actions:
                    success = self.execute_action(action)
                    if success and self.dry_run:
                        # 模拟模式：手动更新持仓
                        pos = self.strategy.positions[action.symbol]
                        price = self.strategy.prices.get(action.symbol, 0)
                        if price > 0:
                            # 根据仓位方向判断加减仓
                            # LONG仓位: BUY=加仓, SELL=减仓
                            # SHORT仓位: SELL=加仓, BUY=减仓
                            if pos.side == PositionSide.LONG:
                                if action.side == "BUY":
                                    new_qty = pos.quantity + action.quantity
                                else:
                                    new_qty = max(0, pos.quantity - action.quantity)
                            else:  # SHORT
                                if action.side == "SELL":
                                    new_qty = pos.quantity + action.quantity
                                else:
                                    new_qty = max(0, pos.quantity - action.quantity)
                            self.strategy.update_position(action.symbol, new_qty, price)

                # 记录调仓
                self.strategy.record_rebalance(actions)
            else:
                logger.info("[调仓] 无需操作")

    def run(self):
        """主循环"""
        self.running = True
        self.start_time = time.time()

        logger.info("="*60)
        logger.info("中性对冲交易器启动")
        logger.info("="*60)
        logger.info(f"模式: {'模拟' if self.dry_run else '实盘'}")
        logger.info(f"订单类型: {'限价单(Maker)' if self.use_limit_orders else '市价单(Taker)'}")
        if self.use_limit_orders:
            logger.info(f"价格偏移: {self.price_offset_pct*100:.3f}%")
        logger.info(f"动态费率: {'开启' if self.dynamic_funding else '关闭'}")
        if self.dynamic_funding:
            logger.info(f"费率检查间隔: {self.funding_check_interval/60:.0f}分钟")
            logger.info(f"结算前检查: 提前{self.pre_settlement_check_minutes}分钟")
        logger.info(f"检查间隔: {self.check_interval}秒")
        logger.info(f"币种配置:")
        for asset in self.strategy.config.assets:
            logger.info(f"  {asset.symbol}: {asset.side.value} ${asset.target_notional:.0f}")
        logger.info("="*60)

        # 启动时立即检查资金费率
        if self.dynamic_funding:
            logger.info("[启动] 检查当前资金费率...")
            self.check_and_update_funding_rates()

        try:
            while self.running:
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"运行错误: {e}")

                # 心跳日志
                runtime = (time.time() - self.start_time) / 60
                logger.info(f"[心跳] 运行 {runtime:.1f}分钟 | "
                           f"调仓 {self.strategy.rebalance_count}次 | "
                           f"交易量 ${self.strategy.total_volume:.0f}")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("收到停止信号，正在退出...")
            self.running = False

        # 打印最终统计
        runtime = (time.time() - self.start_time) / 60
        logger.info("="*60)
        logger.info("交易器停止")
        logger.info(f"运行时间: {runtime:.1f}分钟")
        logger.info(f"调仓次数: {self.strategy.rebalance_count}")
        logger.info(f"总交易量: ${self.strategy.total_volume:.0f}")
        logger.info("="*60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='中性对冲策略交易器')
    parser.add_argument('--dry-run', action='store_true', help='模拟模式 (不实际下单)')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔秒数 (默认60)')
    parser.add_argument('--limit-order', action='store_true', help='使用限价单 (Maker)')
    parser.add_argument('--offset', type=float, default=0.05, help='限价单偏移百分比 (默认0.05%%)')
    parser.add_argument('--dynamic-funding', action='store_true',
                        help='根据资金费率动态调整持仓方向')
    parser.add_argument('--funding-interval', type=int, default=60,
                        help='资金费率检查间隔(分钟，默认60)')
    parser.add_argument('--pre-settlement', type=int, default=5,
                        help='结算前检查时间(分钟，默认5分钟)')
    args = parser.parse_args()

    if not SDK_AVAILABLE:
        raise RuntimeError("apexomni SDK未安装，请运行: pip install apexomni")

    # 从环境变量读取配置
    API_KEY = os.getenv("APEX_API_KEY", "")
    SECRET_KEY = os.getenv("APEX_SECRET_KEY", "")
    PASSPHRASE = os.getenv("APEX_PASSPHRASE", "")
    OMNIKEY = os.getenv("APEX_OMNIKEY", "")
    TESTNET = os.getenv("APEX_TESTNET", "true").lower() == "true"

    # 命令行参数
    DRY_RUN = args.dry_run
    USE_LIMIT_ORDERS = args.limit_order
    PRICE_OFFSET_PCT = args.offset / 100  # 转换为小数
    DYNAMIC_FUNDING = args.dynamic_funding
    FUNDING_CHECK_INTERVAL = args.funding_interval * 60  # 转换为秒
    PRE_SETTLEMENT_MINUTES = args.pre_settlement

    # 策略配置
    # 目标: 150,000U 总仓位 (多头75,000U + 空头75,000U) = Delta中性
    # BTC多$75000 vs ETH空$37500 + SOL空$37500
    # 每次加仓2000U（多头1000U + 空头各500U），1分钟一次
    config = HedgingConfig(
        assets=[
            AssetConfig("BTC-USDT", PositionSide.LONG, 75000.0),
            AssetConfig("ETH-USDT", PositionSide.SHORT, 37500.0),
            AssetConfig("SOL-USDT", PositionSide.SHORT, 37500.0),
        ],
        leverage=3,
        rebalance_interval_seconds=60,  # 1分钟检查一次
        delta_threshold_pct=0.03,  # 3%偏差触发调仓
        min_rebalance_interval_seconds=60,  # 最小1分钟间隔
        max_rebalances_per_hour=60,  # 每小时最多60次
        scale_up_amount=2000.0,  # 每次加仓2000U（多空各1000U）
        rebalance_step_pct=0.05,  # 每次调整5%
        min_trade_notional=100.0,  # 最小交易额
    )

    print("="*60)
    print("Delta中性对冲策略 - 缓慢加仓模式")
    print("="*60)
    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print(f"模式: {'模拟运行' if DRY_RUN else '实盘交易'}")
    print(f"订单类型: {'限价单(Maker)' if USE_LIMIT_ORDERS else '市价单(Taker)'}")
    if USE_LIMIT_ORDERS:
        print(f"价格偏移: {args.offset:.2f}%")
    print(f"动态费率: {'开启' if DYNAMIC_FUNDING else '关闭'}")
    if DYNAMIC_FUNDING:
        print(f"费率检查间隔: {args.funding_interval}分钟")
        print(f"结算前检查: 提前{PRE_SETTLEMENT_MINUTES}分钟")
    print(f"\n目标配置 (总仓位: $150,000):")
    total_long = sum(a.target_notional for a in config.assets if a.side == PositionSide.LONG)
    total_short = sum(a.target_notional for a in config.assets if a.side == PositionSide.SHORT)
    for asset in config.assets:
        print(f"  {asset.symbol}: {asset.side.value} ${asset.target_notional:,.0f}")
    print(f"\n多头总计: ${total_long:,.0f}")
    print(f"空头总计: ${total_short:,.0f}")
    print(f"净Delta: ${total_long - total_short:,.0f} (目标: $0)")
    print(f"\n加仓策略:")
    print(f"  每次加仓: ${config.scale_up_amount:,.0f} (多头${config.scale_up_amount/2:,.0f} + 空头${config.scale_up_amount/2:,.0f})")
    print(f"  加仓间隔: {config.rebalance_interval_seconds}秒 (1分钟)")
    print(f"  预计达到目标: ~{int(150000 / config.scale_up_amount)}分钟 (~{int(150000 / config.scale_up_amount / 60)}小时)")
    print(f"\n风控参数:")
    print(f"  Delta偏差阈值: {config.delta_threshold_pct:.0%}")
    print(f"  最小交易额: ${config.min_trade_notional:.0f}")
    print("="*60)

    if not TESTNET and not DRY_RUN:
        print("\n[!] 警告: 主网实盘交易模式!")
        confirm = input("确认要在主网进行实盘交易? (输入 YES 确认): ")
        if confirm != "YES":
            print("已取消")
            return

    # 初始化
    strategy = NeutralHedgingStrategy(config)
    trader = ApexTrader(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        passphrase=PASSPHRASE,
        omnikey=OMNIKEY,
        testnet=TESTNET
    )

    # 检查账户
    account = trader.get_account()
    if account.get("success"):
        account_id = account.get("data", {}).get("id", "unknown")
        logger.info(f"账户ID: {account_id}")
    else:
        logger.error(f"账户连接失败: {account.get('error')}")
        if not DRY_RUN:
            return

    # 启动交易器
    hedging_trader = HedgingTrader(
        strategy=strategy,
        trader=trader,
        dry_run=DRY_RUN,
        check_interval=args.interval,
        use_limit_orders=USE_LIMIT_ORDERS,
        price_offset_pct=PRICE_OFFSET_PCT,
        dynamic_funding=DYNAMIC_FUNDING,
        funding_check_interval=FUNDING_CHECK_INTERVAL,
        testnet=TESTNET,
        pre_settlement_check_minutes=PRE_SETTLEMENT_MINUTES
    )

    hedging_trader.run()


if __name__ == "__main__":
    main()
