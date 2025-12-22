"""
FOMO策略实盘交易器 V2
使用官方apexomni SDK
支持Apex Exchange主网

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
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fomo_strategy import FOMOStrategy, FOMOStrategyConfig, Signal

# 设置日志 - 强制立即刷新
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
        FlushFileHandler('trading.log', encoding='utf-8'),
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

# 导入官方SDK
try:
    from apexomni.http_public import HttpPublic
    from apexomni.http_private import HttpPrivate
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("apexomni SDK not found. Install with: pip install apexomni")


# 使用Binance获取K线数据（因为Apex公共API不稳定）
import urllib.request
import json
import ssl


def get_binance_klines(symbol: str, interval: str = "1m", limit: int = 100) -> List[dict]:
    """从Binance获取K线数据"""
    # 转换交易对格式: BTC-USDC -> BTCUSDC
    binance_symbol = symbol.replace("-", "")
    if binance_symbol.endswith("USDC"):
        binance_symbol = binance_symbol[:-4] + "USDT"  # Binance没有USDC对，用USDT

    url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"

    ssl_ctx = ssl.create_default_context()
    req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            klines = []
            for k in data:
                klines.append({
                    "t": k[0],  # Open time
                    "o": k[1],  # Open
                    "h": k[2],  # High
                    "l": k[3],  # Low
                    "c": k[4],  # Close
                    "v": k[5],  # Volume
                })
            return klines
    except Exception as e:
        logger.error(f"获取Binance K线失败: {e}")
        return []


class ApexTraderV2:
    """Apex Exchange交易接口 - 使用官方SDK"""

    NETWORK_ID_MAINNET = 42161  # Arbitrum mainnet
    NETWORK_ID_TESTNET = 421614  # Arbitrum Sepolia

    def __init__(self, api_key: str, secret_key: str, passphrase: str,
                 testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.testnet = testnet

        if testnet:
            self.endpoint = "https://testnet.omni.apex.exchange"
            self.network_id = self.NETWORK_ID_TESTNET
        else:
            self.endpoint = "https://omni.apex.exchange"
            self.network_id = self.NETWORK_ID_MAINNET

        # 初始化客户端
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

    def get_account(self) -> dict:
        """获取账户信息"""
        try:
            result = self.private_client.get_account_v2()
            return {"success": True, "data": result.get("data", {})}
        except Exception as e:
            logger.error(f"获取账户失败: {e}")
            return {"success": False, "error": str(e)}

    def get_positions(self) -> List[dict]:
        """获取持仓"""
        result = self.get_account()
        if result.get("success"):
            positions = result.get("data", {}).get("positions", [])
            if positions:
                return [p for p in positions if float(p.get("size", 0)) != 0]
        return []

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """获取行情"""
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
        """获取K线 - 使用Binance数据"""
        return get_binance_klines(symbol, f"{interval}m", limit)

    def place_order(self, symbol: str, side: str, size: float,
                    price: float = None, order_type: str = "MARKET",
                    reduce_only: bool = False) -> dict:
        """下单 - 需要ZK签名，暂时只支持模拟"""
        logger.warning("实际下单需要ZK签名，当前版本暂不支持")
        return {"success": False, "error": "ZK签名未配置"}

    def close_position(self, symbol: str, size: float) -> dict:
        """平仓"""
        return self.place_order(symbol, "SELL", size, reduce_only=True)


class LiveTraderV2:
    """
    实盘交易器 V2

    使用方法:
    1. 配置.env文件中的API密钥
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

        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise ValueError("请在.env中配置APEX_API_KEY, APEX_SECRET_KEY, APEX_PASSPHRASE")

        # 初始化交易接口
        if SDK_AVAILABLE:
            self.trader = ApexTraderV2(
                self.api_key, self.secret_key, self.passphrase, testnet
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
        """更新K线数据"""
        klines = self.trader.get_klines(symbol, "1", 100)
        if not klines:
            logger.warning(f"无法获取{symbol}的K线数据")
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

    def process_signal(self, symbol: str, signal: Signal):
        """处理交易信号"""
        if signal.action == "NONE":
            return

        logger.info(f"[{symbol}] 信号: {signal.action} - {signal.reason}")

        if signal.action == "OPEN_LONG":
            if self.dry_run:
                logger.info(f"[模拟] 开多 {symbol}: {signal.quantity:.6f} @ {signal.price:.2f}")
                logger.info(f"[模拟] 止损: {signal.stop_price:.2f}")
            else:
                result = self.trader.place_order(
                    symbol=symbol,
                    side="BUY",
                    size=signal.quantity,
                    order_type="MARKET"
                )
                if result.get("success"):
                    logger.info(f"开仓成功: {result}")
                    self.strategies[symbol].on_trade_executed(
                        symbol, "OPEN_LONG", signal.price, signal.quantity,
                        datetime.now()
                    )
                else:
                    logger.error(f"开仓失败: {result}")

        elif signal.action == "CLOSE":
            strategy = self.strategies[symbol]
            if strategy.position:
                if self.dry_run:
                    logger.info(f"[模拟] 平仓 {symbol}: {strategy.position.quantity:.6f} @ {signal.price:.2f}")
                else:
                    result = self.trader.close_position(
                        symbol=symbol,
                        size=strategy.position.quantity
                    )
                    if result.get("success"):
                        logger.info(f"平仓成功: {result}")
                        strategy.on_trade_executed(
                            symbol, "CLOSE", signal.price,
                            strategy.position.quantity, datetime.now()
                        )
                    else:
                        logger.error(f"平仓失败: {result}")

    def run_once(self):
        """运行一次检查"""
        logger.info(f"--- 开始检查 ({datetime.now().strftime('%H:%M:%S')}) ---")
        for symbol in self.symbols:
            try:
                logger.info(f"[{symbol}] 获取K线数据...")
                # 更新数据
                if not self.update_klines(symbol):
                    continue

                logger.info(f"[{symbol}] 生成信号...")
                # 生成信号
                signal = self.strategies[symbol].generate_signal(symbol)

                # 处理信号
                self.process_signal(symbol, signal)

                # 输出状态
                status = self.strategies[symbol].get_status()
                if status["has_position"]:
                    pos = status["position"]
                    logger.info(f"[{symbol}] 持仓中: {pos['quantity']:.6f} @ {pos['entry_price']:.2f}, "
                               f"止损: {pos['stop_price']:.2f}, 追踪: {pos['trail_stop']}")
                else:
                    # 显示当前价格
                    if len(self.strategies[symbol].closes) > 0:
                        current_price = self.strategies[symbol].closes[-1]
                        logger.info(f"[{symbol}] 当前价格: {current_price:.2f}, 信号: {signal.reason}")

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

        try:
            while self.running:
                self.run_once()
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
    # 配置
    SYMBOLS = ["BTC-USDC", "ETH-USDC"]  # 交易对
    TESTNET = os.getenv("APEX_TESTNET", "true").lower() == "true"
    DRY_RUN = True  # 设为False进行实盘交易

    print("=" * 50)
    print("FOMO策略实盘交易器 V2")
    print("=" * 50)
    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print(f"模式: {'模拟运行' if DRY_RUN else '实盘交易'}")
    print(f"交易对: {SYMBOLS}")
    print("=" * 50)

    if not TESTNET and not DRY_RUN:
        print("\n警告: 主网实盘交易模式!")
        print("请确保:")
        print("1. API密钥配置正确")
        print("2. 账户有足够资金")
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
