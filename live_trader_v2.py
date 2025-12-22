"""
FOMO策略实盘交易器 V2
使用官方apexomni SDK + OMNIKEY实现真实下单
支持Apex Exchange主网

数据源: Binance (更活跃的成交量数据)
交易所: Apex Exchange (下单执行)

安全特性:
1. 启动前确认模式
2. 最大持仓限制
3. 紧急停止功能
4. 详细日志记录

使用Python 3.12运行:
  py -3.12 live_trader_v2.py
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

from fomo_strategy import FOMOStrategy, FOMOStrategyConfig, Signal

# 设置日志 - 强制立即刷新，保存到项目目录
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, 'trading.log')

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


def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 100) -> List[dict]:
    """
    从Binance获取K线数据

    Args:
        symbol: 交易对 (如 BTC-USDT)
        interval: K线间隔 (1m, 5m, 15m, 1h, etc.)
        limit: 获取数量
    """
    # 转换交易对格式: BTC-USDT -> BTCUSDT, 1000PEPE-USDT -> PEPEUSDT
    binance_symbol = symbol.replace("-", "")
    if binance_symbol.startswith("1000"):
        binance_symbol = binance_symbol[4:]

    ssl_ctx = ssl.create_default_context()
    url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            klines = []
            for k in data:
                klines.append({
                    "t": k[0],  # Open time
                    "o": float(k[1]),
                    "h": float(k[2]),
                    "l": float(k[3]),
                    "c": float(k[4]),
                    "v": float(k[5]),
                })
            return klines
    except Exception as e:
        logger.error(f"获取Binance K线失败 {symbol}: {e}")
        return []


# 导入官方SDK
try:
    from apexomni.http_public import HttpPublic
    from apexomni.http_private import HttpPrivate
    from apexomni.http_private_sign import HttpPrivateSign
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("apexomni SDK not found. Install with: pip install apexomni")


class ApexTraderV2:
    """Apex Exchange交易接口 - 使用官方SDK + OMNIKEY"""

    NETWORK_ID_MAINNET = 42161  # Arbitrum mainnet
    NETWORK_ID_TESTNET = 421614  # Arbitrum Sepolia

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

        # 初始化公共客户端
        self.public_client = HttpPublic(self.endpoint)

        # 初始化私有客户端 (用于查询)
        self.private_client = HttpPrivate(
            self.endpoint,
            network_id=self.network_id,
            api_key_credentials={
                'key': api_key,
                'secret': secret_key,
                'passphrase': passphrase
            }
        )

        # 初始化签名客户端 (用于下单) - 使用OMNIKEY
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
                # 加载配置（下单前需要）
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

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """获取行情 - 使用Apex数据"""
        try:
            result = self.public_client.ticker_v3(symbol=symbol)
            if result.get("data"):
                for t in result.get("data", []):
                    if t.get("symbol") == symbol:
                        return t
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
        return None

    def get_klines(self, symbol: str, interval: str = "1",
                   limit: int = 100) -> List[dict]:
        """获取K线 - 使用Apex数据"""
        try:
            # Apex kline API: interval为分钟数
            result = self.public_client.klines_v3(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            if result.get("data"):
                # Apex格式: {'data': {'BTCUSDT': [...]}}
                # symbol格式转换: BTC-USDT -> BTCUSDT
                apex_symbol = symbol.replace("-", "")
                data = result.get("data", {})

                # data可能是dict或list
                if isinstance(data, dict):
                    kline_list = data.get(apex_symbol, [])
                else:
                    kline_list = data

                klines = []
                for k in kline_list:
                    klines.append({
                        "t": k.get("t", 0),  # Open time
                        "o": k.get("o", "0"),  # Open
                        "h": k.get("h", "0"),  # High
                        "l": k.get("l", "0"),  # Low
                        "c": k.get("c", "0"),  # Close
                        "v": k.get("v", "0"),  # Volume
                    })
                return klines
        except Exception as e:
            logger.error(f"获取Apex K线失败: {e}")
        return []

    def _round_price_to_tick(self, price: float, symbol: str) -> float:
        """将价格圆整到tickSize的倍数"""
        try:
            # 从configV3获取tickSize
            if self.sign_client and hasattr(self.sign_client, 'configV3') and self.sign_client.configV3:
                config = self.sign_client.configV3
                contracts = config.get('contractConfig', {}).get('perpetualContract', [])
                for c in contracts:
                    if c.get('symbol') == symbol or c.get('symbolDisplayName') == symbol:
                        tick_size = float(c.get('tickSize', 0.01))
                        # 圆整到tickSize的倍数
                        rounded = round(price / tick_size) * tick_size
                        # 保留合适的小数位数
                        decimals = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
                        return round(rounded, decimals)
        except Exception as e:
            logger.warning(f"获取tickSize失败: {e}")
        return price

    def _round_size_to_step(self, size: float, symbol: str) -> float:
        """将数量圆整到stepSize的倍数"""
        try:
            if self.sign_client and hasattr(self.sign_client, 'configV3') and self.sign_client.configV3:
                config = self.sign_client.configV3
                contracts = config.get('contractConfig', {}).get('perpetualContract', [])
                for c in contracts:
                    if c.get('symbol') == symbol or c.get('symbolDisplayName') == symbol:
                        step_size = float(c.get('stepSize', 0.001))
                        # 向下圆整到stepSize的倍数
                        rounded = int(size / step_size) * step_size
                        decimals = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
                        return round(rounded, decimals)
        except Exception as e:
            logger.warning(f"获取stepSize失败: {e}")
        return size

    def place_order(self, symbol: str, side: str, size: float,
                    price: float = None, order_type: str = "MARKET",
                    reduce_only: bool = False) -> dict:
        """下单 - 使用OMNIKEY签名"""
        if not self.sign_client:
            logger.error("OMNIKEY未配置，无法下单")
            return {"success": False, "error": "OMNIKEY未配置"}

        try:
            # 圆整价格和数量
            if price:
                price = self._round_price_to_tick(price, symbol)
            size = self._round_size_to_step(size, symbol)

            # 构建订单参数
            order_params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "size": str(size),
                "timestampSeconds": time.time(),
                "reduceOnly": reduce_only
            }

            # apexomni SDK要求price参数，即使是MARKET单也需要
            if price:
                order_params["price"] = str(price)
            else:
                # 如果没有传price，尝试获取当前市场价
                try:
                    ticker = self.public_client.ticker_v3(symbol=symbol)
                    if ticker.get("data"):
                        ticker_data = ticker["data"]
                        if isinstance(ticker_data, list):
                            current_price = float(ticker_data[0].get("lastPrice", 0))
                        else:
                            current_price = float(ticker_data.get("lastPrice", 0))
                        if current_price > 0:
                            current_price = self._round_price_to_tick(current_price, symbol)
                            order_params["price"] = str(current_price)
                            logger.debug(f"MARKET单使用当前价格: {current_price}")
                except Exception as e:
                    logger.warning(f"获取ticker失败，尝试继续: {e}")

            logger.debug(f"下单参数: {order_params}")
            result = self.sign_client.create_order_v3(**order_params)

            if result.get("data"):
                order_id = result.get("data", {}).get("orderId")
                logger.info(f"下单成功: {order_id}")
                return {"success": True, "data": result.get("data")}
            else:
                logger.error(f"下单失败: {result}")
                return {"success": False, "error": str(result)}

        except Exception as e:
            logger.error(f"下单异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def close_position(self, symbol: str, size: float, side: str = "SELL",
                       price: float = None) -> dict:
        """平仓"""
        return self.place_order(
            symbol=symbol,
            side=side,
            size=size,
            price=price,  # SDK需要price参数
            order_type="MARKET",
            reduce_only=True
        )

    def cancel_order(self, order_id: str) -> dict:
        """取消订单"""
        if not self.sign_client:
            return {"success": False, "error": "OMNIKEY未配置"}

        try:
            result = self.sign_client.delete_order_v3(id=order_id)
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return {"success": False, "error": str(e)}


class LiveTraderV2:
    """
    实盘交易器 V2

    使用方法:
    1. 配置.env文件中的API密钥和OMNIKEY
    2. 设置APEX_TESTNET=false使用主网
    3. 运行 py -3.12 live_trader_v2.py
    """

    def __init__(self, symbols: List[str], testnet: bool = True,
                 dry_run: bool = True):
        """
        初始化

        Args:
            symbols: 交易对列表
            testnet: 是否使用测试网
            dry_run: 模拟运行(不实际下单)
        """
        self.symbols = symbols
        self.testnet = testnet
        self.dry_run = dry_run

        # 加载API配置
        self.api_key = os.getenv("APEX_API_KEY", "")
        self.secret_key = os.getenv("APEX_SECRET_KEY", "")
        self.passphrase = os.getenv("APEX_PASSPHRASE", "")
        self.omnikey = os.getenv("OMNIKEY", "")

        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise ValueError("请在.env中配置APEX_API_KEY, APEX_SECRET_KEY, APEX_PASSPHRASE")

        if not dry_run and not self.omnikey:
            raise ValueError("实盘交易需要在.env中配置OMNIKEY")

        # 初始化交易接口
        if SDK_AVAILABLE:
            self.trader = ApexTraderV2(
                self.api_key, self.secret_key, self.passphrase,
                omnikey=self.omnikey, testnet=testnet
            )
        else:
            raise RuntimeError("apexomni SDK未安装，请运行: pip install apexomni")

        # 初始化策略(每个品种一个)
        self.strategies: Dict[str, FOMOStrategy] = {}
        for symbol in symbols:
            config = FOMOStrategyConfig()
            self.strategies[symbol] = FOMOStrategy(config)

        # 运行状态
        self.running = False
        self.last_check_time: Dict[str, datetime] = {}

        logger.info(f"LiveTraderV2初始化完成")
        logger.info(f"  网络: {'测试网' if testnet else '主网'}")
        logger.info(f"  模式: {'模拟' if dry_run else '实盘'}")
        logger.info(f"  交易对: {symbols}")

    def check_account(self) -> bool:
        """检查账户状态"""
        result = self.trader.get_account()
        if not result.get("success"):
            logger.error(f"无法获取账户: {result}")
            return False

        data = result.get("data", {})
        logger.info(f"账户状态:")
        logger.info(f"  账户ID: {data.get('id', 'N/A')}")

        # 检查现有持仓
        positions = self.trader.get_positions()
        if positions:
            logger.info(f"  当前持仓: {len(positions)}")
            for p in positions:
                logger.info(f"    {p.get('symbol')}: {p.get('size')} @ {p.get('entryPrice')}")

        return True

    def update_klines(self, symbol: str) -> bool:
        """更新K线数据 - 使用Binance数据源"""
        # 使用Binance获取K线 (成交量数据更活跃)
        klines = fetch_binance_klines(symbol, "1m", 100)
        if not klines:
            logger.warning(f"无法获取{symbol}的Binance K线数据")
            return False

        strategy = self.strategies[symbol]

        for k in klines:
            try:
                ts = k.get("t", 0)
                if isinstance(ts, str):
                    ts = int(ts)
                dt = datetime.fromtimestamp(ts / 1000)

                strategy.update(
                    timestamp=dt,
                    open_=float(k.get("o", 0)),
                    high=float(k.get("h", 0)),
                    low=float(k.get("l", 0)),
                    close=float(k.get("c", 0)),
                    volume=float(k.get("v", 0))
                )
            except Exception as e:
                logger.error(f"解析K线错误: {e}")

        return True

    def _get_apex_price(self, symbol: str) -> Optional[float]:
        """获取Apex当前价格用于下单"""
        ticker = self.trader.get_ticker(symbol)
        if ticker:
            return float(ticker.get("lastPrice", 0))
        return None

    def _count_open_positions(self) -> int:
        """统计当前持仓数量"""
        count = 0
        for strategy in self.strategies.values():
            if strategy.position is not None:
                count += 1
        return count

    def process_signal(self, symbol: str, signal: Signal):
        """处理交易信号 (支持多空双向)"""
        if signal.action == "NONE":
            return

        logger.info(f"[{symbol}] 信号: {signal.action} - {signal.reason}")

        # 检查最大持仓数限制 (仅对开仓信号)
        if signal.action in ["OPEN_LONG", "OPEN_SHORT"]:
            current_positions = self._count_open_positions()
            max_positions = self.strategies[symbol].config.max_positions
            if current_positions >= max_positions:
                logger.warning(f"[{symbol}] 跳过开仓: 已达最大持仓数 {current_positions}/{max_positions}")
                return

        # 获取Apex实时价格用于下单 (信号来自Binance，但下单在Apex执行)
        apex_price = self._get_apex_price(symbol)
        if not apex_price:
            logger.warning(f"[{symbol}] 无法获取Apex价格，使用信号价格")
            apex_price = signal.price

        if signal.action == "OPEN_LONG":
            if self.dry_run:
                logger.info(f"[模拟] 开多 {symbol}: {signal.quantity:.6f} @ {apex_price:.2f}")
                logger.info(f"[模拟] 止损: {signal.stop_price:.2f}")
            else:
                result = self.trader.place_order(
                    symbol=symbol,
                    side="BUY",
                    size=signal.quantity,
                    price=apex_price,  # 使用Apex价格
                    order_type="MARKET"
                )
                if result.get("success"):
                    logger.info(f"开多成功: {result}")
                    self.strategies[symbol].on_trade_executed(
                        symbol, "OPEN_LONG", apex_price, signal.quantity,
                        datetime.now()
                    )
                else:
                    logger.error(f"开多失败: {result}")

        elif signal.action == "OPEN_SHORT":
            if self.dry_run:
                logger.info(f"[模拟] 开空 {symbol}: {signal.quantity:.6f} @ {apex_price:.2f}")
                logger.info(f"[模拟] 止损: {signal.stop_price:.2f}")
            else:
                result = self.trader.place_order(
                    symbol=symbol,
                    side="SELL",
                    size=signal.quantity,
                    price=apex_price,  # 使用Apex价格
                    order_type="MARKET"
                )
                if result.get("success"):
                    logger.info(f"开空成功: {result}")
                    self.strategies[symbol].on_trade_executed(
                        symbol, "OPEN_SHORT", apex_price, signal.quantity,
                        datetime.now()
                    )
                else:
                    logger.error(f"开空失败: {result}")

        elif signal.action == "CLOSE":
            strategy = self.strategies[symbol]
            if strategy.position:
                # 根据持仓方向决定平仓方向
                close_side = "SELL" if strategy.position.side == "LONG" else "BUY"
                side_text = "平多" if strategy.position.side == "LONG" else "平空"

                if self.dry_run:
                    logger.info(f"[模拟] {side_text} {symbol}: {strategy.position.quantity:.6f} @ {apex_price:.2f}")
                else:
                    result = self.trader.close_position(
                        symbol=symbol,
                        size=strategy.position.quantity,
                        side=close_side,
                        price=apex_price  # 使用Apex价格
                    )
                    if result.get("success"):
                        logger.info(f"{side_text}成功: {result}")
                        strategy.on_trade_executed(
                            symbol, "CLOSE", signal.price,
                            strategy.position.quantity, datetime.now()
                        )
                    else:
                        logger.error(f"{side_text}失败: {result}")

    def run_once(self):
        """运行一次检查"""
        logger.debug(f"--- 开始检查 ({datetime.now().strftime('%H:%M:%S')}) ---")
        for symbol in self.symbols:
            try:
                # 更新数据
                if not self.update_klines(symbol):
                    continue

                # 生成信号
                signal = self.strategies[symbol].generate_signal(symbol)

                # 处理信号
                self.process_signal(symbol, signal)

                # 只输出持仓状态（重要信息）
                status = self.strategies[symbol].get_status()
                if status["has_position"]:
                    pos = status["position"]
                    logger.info(f"[{symbol}] 持仓中: {pos['quantity']:.6f} @ {pos['entry_price']:.2f}, "
                               f"止损: {pos['stop_price']:.2f}, 追踪: {pos['trail_stop']}")

            except Exception as e:
                logger.error(f"处理{symbol}时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def run(self, interval_seconds: int = 60):
        """
        运行交易循环

        Args:
            interval_seconds: 检查间隔(秒)
        """
        logger.info("=" * 50)
        logger.info("FOMO策略实盘交易启动 (V2)")
        logger.info("=" * 50)

        # 检查账户
        if not self.check_account():
            logger.error("账户检查失败，退出")
            return

        # 确认启动
        if not self.testnet and not self.dry_run:
            logger.warning("=" * 50)
            logger.warning("警告: 即将在主网进行实盘交易!")
            logger.warning("=" * 50)
            confirm = input("确认启动? (输入 YES 确认): ")
            if confirm != "YES":
                logger.info("用户取消启动")
                return

        self.running = True
        logger.info(f"开始运行，检查间隔: {interval_seconds}秒")
        logger.info("按 Ctrl+C 停止")

        heartbeat_interval = 3  # 每3分钟输出一次心跳
        loop_count = 0

        try:
            while self.running:
                self.run_once()
                loop_count += 1

                # 每3分钟输出心跳状态
                if loop_count % heartbeat_interval == 0:
                    positions = sum(1 for s in self.symbols if self.strategies[s].get_status()["has_position"])
                    logger.info(f"[心跳] 运行中 | 扫描{len(self.symbols)}币种 | 当前持仓: {positions}个")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("收到停止信号，正在退出...")
            self.running = False

        # 输出最终状态
        logger.info("=" * 50)
        logger.info("交易结束，最终状态:")
        for symbol in self.symbols:
            status = self.strategies[symbol].get_status()
            logger.info(f"[{symbol}]")
            logger.info(f"  交易次数: {status['total_trades']}")
            logger.info(f"  胜率: {status['win_rate']:.1%}")
            logger.info(f"  总盈亏: {status['total_pnl']:.2f}")


def main():
    """主函数"""
    # 配置 - Binance和Apex都有的交易对 (移除Apex特有的新币)
    SYMBOLS = [
        # 主流币 (最高流动性)
        "BTC-USDT", "ETH-USDT", "SOL-USDT",
        # 大盘币
        "AAVE-USDT", "LINK-USDT", "UNI-USDT", "BCH-USDT", "LTC-USDT",
        "NEAR-USDT", "APT-USDT", "INJ-USDT", "ARB-USDT", "OP-USDT",
        # Meme币 (高波动) - Binance用PEPE/BONK不带1000前缀，但代码会自动转换
        "1000PEPE-USDT", "1000BONK-USDT", "WIF-USDT",
        # DeFi
        "PENDLE-USDT", "CRV-USDT", "LDO-USDT", "JUP-USDT", "ENA-USDT",
        # L2/基础设施
        "STX-USDT", "TIA-USDT", "EIGEN-USDT",
        # 其他高交易量 (Binance都有)
        "ORDI-USDT", "WLD-USDT", "AR-USDT", "RENDER-USDT",
        "ZEN-USDT", "HBAR-USDT", "ENS-USDT", "CAKE-USDT",
        # 主流补充
        "DOGE-USDT", "XRP-USDT", "ADA-USDT", "AVAX-USDT", "DOT-USDT",
    ]
    TESTNET = os.getenv("APEX_TESTNET", "true").lower() == "true"
    DRY_RUN = False  # 实盘交易

    print("=" * 50)
    print("FOMO策略实盘交易器 V2")
    print("=" * 50)
    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print(f"模式: {'模拟运行' if DRY_RUN else '实盘交易'}")
    print(f"交易对: {SYMBOLS}")
    print(f"K线数据: Binance (成交量更活跃)")
    print(f"交易执行: Apex Exchange")
    print("=" * 50)

    if not TESTNET and not DRY_RUN:
        print("\n警告: 主网实盘交易模式!")
        print("请确保:")
        print("1. API密钥和OMNIKEY配置正确")
        print("2. 账户有足够资金 (USDT)")
        print("3. 已了解策略风险")
        print()

    trader = LiveTraderV2(
        symbols=SYMBOLS,
        testnet=TESTNET,
        dry_run=DRY_RUN
    )

    trader.run(interval_seconds=60)


if __name__ == "__main__":
    main()
