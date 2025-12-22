"""
FOMO策略实盘交易器
支持Apex Exchange主网

安全特性:
1. 启动前确认模式
2. 最大持仓限制
3. 紧急停止功能
4. 详细日志记录
"""
import os
import sys
import time
import json
import hmac
import base64
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
import urllib.request
import ssl

from fomo_strategy import FOMOStrategy, FOMOStrategyConfig, Signal

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading.log', encoding='utf-8'),
        logging.StreamHandler()
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


class ApexTrader:
    """Apex Exchange交易接口"""

    REST_TESTNET = "https://testnet.omni.apex.exchange/api"
    REST_MAINNET = "https://omni.apex.exchange/api"

    def __init__(self, api_key: str, secret_key: str, passphrase: str,
                 testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = self.REST_TESTNET if testnet else self.REST_MAINNET
        self.testnet = testnet
        self.ssl_context = ssl.create_default_context()

    def _sign(self, timestamp: str, method: str, request_path: str,
              body: str = "") -> str:
        """生成签名"""
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _request(self, method: str, endpoint: str, params: dict = None,
                 body: dict = None, need_auth: bool = True) -> dict:
        """发送请求"""
        request_path = endpoint
        if params:
            query = urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                request_path = f"{endpoint}?{query}"

        url = f"{self.base_url}{request_path}"

        if need_auth:
            timestamp = str(int(time.time()))
            body_str = json.dumps(body) if body else ""
            sign = self._sign(timestamp, method, request_path, body_str)
            headers = {
                "APEX-API-KEY": self.api_key,
                "APEX-API-SIGNATURE": sign,
                "APEX-API-TIMESTAMP": timestamp,
                "APEX-API-PASSPHRASE": self.passphrase,
                "Content-Type": "application/json" if body else "application/x-www-form-urlencoded",
            }
        else:
            headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(url, method=method, headers=headers)
        if body:
            req.data = json.dumps(body).encode('utf-8')

        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return {"success": False, "error": str(e)}

    def get_account(self) -> dict:
        """获取账户信息"""
        return self._request("GET", "/v3/account")

    def get_positions(self) -> List[dict]:
        """获取持仓"""
        result = self._request("GET", "/v3/account")
        if result.get("success"):
            positions = result.get("data", {}).get("positions", [])
            return [p for p in positions if float(p.get("size", 0)) != 0]
        return []

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """获取行情"""
        result = self._request("GET", "/v3/public/ticker",
                               {"symbol": symbol}, need_auth=False)
        if result.get("success"):
            for t in result.get("data", []):
                if t.get("symbol") == symbol:
                    return t
        return None

    def get_klines(self, symbol: str, interval: str = "1",
                   limit: int = 100) -> List[dict]:
        """获取K线"""
        result = self._request("GET", "/v3/public/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }, need_auth=False)

        if result.get("success"):
            return result.get("data", [])
        return []

    def place_order(self, symbol: str, side: str, size: float,
                    price: float = None, order_type: str = "MARKET",
                    reduce_only: bool = False) -> dict:
        """下单"""
        body = {
            "symbol": symbol,
            "side": side,  # "BUY" or "SELL"
            "type": order_type,
            "size": str(size),
            "reduceOnly": reduce_only,
        }
        if price and order_type == "LIMIT":
            body["price"] = str(price)

        logger.info(f"下单: {body}")
        return self._request("POST", "/v3/order", body=body)

    def close_position(self, symbol: str, size: float) -> dict:
        """平仓"""
        return self.place_order(symbol, "SELL", size, reduce_only=True)


class LiveTrader:
    """
    实盘交易器

    使用方法:
    1. 配置.env文件中的API密钥
    2. 设置APEX_TESTNET=false使用主网
    3. 运行 python live_trader.py
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
        self.trader = ApexTrader(
            self.api_key, self.secret_key, self.passphrase, testnet
        )

        # 初始化策略(每个品种一个)
        self.strategies: Dict[str, FOMOStrategy] = {}
        for symbol in symbols:
            config = FOMOStrategyConfig()
            self.strategies[symbol] = FOMOStrategy(config)

        # 运行状态
        self.running = False
        self.last_check_time: Dict[str, datetime] = {}

        logger.info(f"LiveTrader初始化完成")
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
        equity = float(data.get("totalEquity", 0))
        available = float(data.get("availableBalance", 0))

        logger.info(f"账户状态:")
        logger.info(f"  总权益: {equity:.2f} USDC")
        logger.info(f"  可用余额: {available:.2f} USDC")

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
        for symbol in self.symbols:
            try:
                # 更新数据
                if not self.update_klines(symbol):
                    continue

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

            except Exception as e:
                logger.error(f"处理{symbol}时出错: {e}")

    def run(self, interval_seconds: int = 60):
        """
        运行交易循环

        Args:
            interval_seconds: 检查间隔(秒)
        """
        logger.info("=" * 50)
        logger.info("FOMO策略实盘交易启动")
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
    print("FOMO策略实盘交易器")
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

    trader = LiveTrader(
        symbols=SYMBOLS,
        testnet=TESTNET,
        dry_run=DRY_RUN
    )

    trader.run(interval_seconds=60)


if __name__ == "__main__":
    main()
