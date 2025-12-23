"""
波动率自适应交易执行器 - Volatility Adaptive Trader
===================================================

执行波动率自适应Delta中性策略：
- 高波动率自动降低仓位 (75000U -> 25000U)
- 低波动率自动增加仓位 (25000U -> 75000U)
- ETH/SOL按波动率反比分配空头占比
- 1分钟一次缓慢调仓

使用方法:
  py -3.12 volatility_adaptive_trader.py --interval 60 --limit-order
"""
import os
import sys
import time
import json
import ssl
import urllib.request
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

from volatility_adaptive_strategy import (
    VolatilityAdaptiveStrategy, AdaptiveHedgingConfig,
    VolatilityConfig, PositionSide, RebalanceAction
)

# 设置日志 - 保存在项目目录的logs子目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'volatility_trader.log')


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
    ssl_ctx = ssl.create_default_context()
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return float(data.get("price", 0))
    except Exception as e:
        logger.error(f"获取价格失败 {symbol}: {e}")
        return None


# 导入交易SDK
try:
    from apexomni.http_public import HttpPublic
    from apexomni.http_private import HttpPrivate
    from apexomni.http_private_sign import HttpPrivateSign
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("apexomni SDK not found")


class ApexTrader:
    """Apex交易接口 (复用hedging_trader中的实现)"""

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
                logger.error(f"OMNIKEY初始化失败: {e}")
                self.sign_client = None

    def get_account(self) -> dict:
        try:
            if self.sign_client:
                result = self.sign_client.get_account_v3()
            else:
                result = self.private_client.get_account_v2()
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_positions(self) -> List[dict]:
        result = self.get_account()
        if result.get("success"):
            positions = result.get("data", {}).get("positions", [])
            return [p for p in positions if float(p.get("size", 0)) != 0]
        return []

    def _round_size_to_step(self, size: float, symbol: str) -> float:
        try:
            if self.sign_client and hasattr(self.sign_client, 'configV3'):
                config = self.sign_client.configV3
                contracts = config.get('contractConfig', {}).get('perpetualContract', [])
                for c in contracts:
                    if c.get('symbol') == symbol or c.get('symbolDisplayName') == symbol:
                        step_size = float(c.get('stepSize', 0.001))
                        rounded = int(size / step_size) * step_size
                        decimals = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
                        return round(rounded, decimals)
        except:
            pass
        return size

    def _round_price_to_tick(self, price: float, symbol: str) -> float:
        try:
            if self.sign_client and hasattr(self.sign_client, 'configV3'):
                config = self.sign_client.configV3
                contracts = config.get('contractConfig', {}).get('perpetualContract', [])
                for c in contracts:
                    if c.get('symbol') == symbol or c.get('symbolDisplayName') == symbol:
                        tick_size = float(c.get('tickSize', 0.01))
                        rounded = round(price / tick_size) * tick_size
                        decimals = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
                        return round(rounded, decimals)
        except:
            pass
        return price

    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "MARKET", price: float = None,
                    reduce_only: bool = False) -> dict:
        if not self.sign_client:
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

            if order_type.upper() == "LIMIT":
                if price is None:
                    return {"success": False, "error": "限价单必须指定价格"}
                price = self._round_price_to_tick(price, symbol)
                order_params["price"] = str(price)
            else:
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

    def get_order_status(self, order_id: str) -> Optional[dict]:
        try:
            if self.sign_client:
                result = self.sign_client.get_order_v3(id=order_id)
                return result.get("data", {})
        except:
            pass
        return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            if self.sign_client:
                self.sign_client.delete_order_v3(id=order_id)
                return True
        except:
            pass
        return False


class VolatilityAdaptiveTrader:
    """波动率自适应交易器"""

    def __init__(self, strategy: VolatilityAdaptiveStrategy, trader: ApexTrader,
                 dry_run: bool = True, check_interval: int = 60,
                 rebalance_interval: int = 300,  # 调仓间隔5分钟
                 use_limit_orders: bool = True, price_offset_pct: float = 0.0005,
                 testnet: bool = True):
        self.strategy = strategy
        self.trader = trader
        self.dry_run = dry_run
        self.check_interval = check_interval  # 挂单检查间隔（1分钟）
        self.rebalance_interval = rebalance_interval  # 调仓间隔（5分钟）
        self.use_limit_orders = use_limit_orders
        self.price_offset_pct = price_offset_pct
        self.testnet = testnet
        self.running = False

        self.pending_orders: Dict[str, dict] = {}
        self.trades_executed = 0
        self.start_time = None
        self.last_rebalance_check = 0  # 上次调仓检查时间

    def sync_positions(self):
        """同步持仓"""
        positions = self.trader.get_positions()

        for symbol in self.strategy.positions.keys():
            found = False
            for p in positions:
                if p.get("symbol") == symbol:
                    size = abs(float(p.get("size", 0)))
                    entry_price = float(p.get("entryPrice", 0))
                    self.strategy.update_position(symbol, size, entry_price)
                    found = True
                    break

            if not found:
                self.strategy.update_position(symbol, 0, 0)

    def update_prices(self):
        """更新价格"""
        for symbol in self.strategy.positions.keys():
            price = fetch_binance_price(symbol)
            if price:
                self.strategy.update_price(symbol, price)

    def check_pending_orders(self, max_wait_seconds: int = 15):
        """检查挂单状态"""
        if not self.pending_orders or self.dry_run:
            return

        current_time = time.time()
        orders_to_remove = []
        orders_to_retry = []

        for order_id, order_info in list(self.pending_orders.items()):
            elapsed = current_time - order_info["time"]
            status = self.trader.get_order_status(order_id)

            if status:
                order_status = status.get("status", "")

                if order_status == "FILLED":
                    logger.info(f"[订单成交] {order_info['symbol']} {order_info['side']}")
                    orders_to_remove.append(order_id)
                elif order_status == "CANCELED":
                    orders_to_remove.append(order_id)
                elif order_status in ["PENDING", "OPEN"]:
                    if elapsed > max_wait_seconds:
                        orders_to_retry.append((order_id, order_info))

        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

        for order_id, order_info in orders_to_retry:
            if self.trader.cancel_order(order_id):
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]

                retry_count = order_info.get("retry_count", 0) + 1

                if retry_count > 3:
                    logger.warning(f"[市价成交] {order_info['symbol']}")
                    self.trader.place_order(
                        symbol=order_info["symbol"],
                        side=order_info["side"],
                        size=order_info["quantity"],
                        order_type="MARKET"
                    )
                    continue

                current_price = fetch_binance_price(order_info["symbol"])
                if current_price:
                    offsets = {1: 0.0003, 2: 0.0001, 3: 0.00005}
                    adjusted_offset = offsets.get(retry_count, 0.00005)

                    if order_info["side"] == "BUY":
                        new_price = current_price * (1 - adjusted_offset)
                    else:
                        new_price = current_price * (1 + adjusted_offset)

                    result = self.trader.place_order(
                        symbol=order_info["symbol"],
                        side=order_info["side"],
                        size=order_info["quantity"],
                        order_type="LIMIT",
                        price=new_price
                    )

                    if result.get("success"):
                        new_order_id = result.get("order_id")
                        self.pending_orders[new_order_id] = {
                            **order_info,
                            "price": new_price,
                            "time": time.time(),
                            "retry_count": retry_count
                        }

    def execute_action(self, action: RebalanceAction) -> bool:
        """执行调仓操作"""
        if self.dry_run:
            logger.info(f"[模拟] {action}")
            return True

        current_price = self.strategy.prices.get(action.symbol, 0)
        if current_price <= 0:
            current_price = fetch_binance_price(action.symbol)

        if self.use_limit_orders and current_price > 0:
            if action.side == "BUY":
                limit_price = current_price * (1 - self.price_offset_pct)
            else:
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
                logger.info(f"[限价单] {action.symbol} {action.side} @ {limit_price:.2f}")
                self.pending_orders[order_id] = {
                    "symbol": action.symbol,
                    "side": action.side,
                    "quantity": action.quantity,
                    "price": limit_price,
                    "notional": action.notional,
                    "time": time.time()
                }
                self.trades_executed += 1
                return True
        else:
            result = self.trader.place_order(
                symbol=action.symbol,
                side=action.side,
                size=action.quantity,
                order_type="MARKET"
            )
            if result.get("success"):
                logger.info(f"[市价单] {action}")
                self.trades_executed += 1
                return True

        return False

    def run_once(self):
        """运行一次（每分钟调用）"""
        current_time = time.time()

        # 1. 每次都检查挂单状态（每分钟）
        if self.use_limit_orders and not self.dry_run:
            self.check_pending_orders(max_wait_seconds=15)
            if self.pending_orders:
                logger.info(f"[挂单] 待成交: {len(self.pending_orders)} 单")

        # 2. 更新价格和持仓
        self.update_prices()
        if not self.dry_run:
            self.sync_positions()

        # 3. 检查是否到达调仓时间（每5分钟）
        time_since_last = current_time - self.last_rebalance_check
        if time_since_last < self.rebalance_interval:
            remaining = self.rebalance_interval - time_since_last
            logger.info(f"[等待] 距下次调仓检查: {remaining:.0f}秒")
            return

        # 4. 调仓检查时间到，执行完整检查
        self.last_rebalance_check = current_time

        # 获取状态
        status = self.strategy.get_status()

        # 打印波动率信息
        logger.info(f"[波动率] " + " | ".join([
            f"{s}: {v*100:.2f}%" for s, v in status['volatilities'].items()
        ]))

        # 打印仓位状态
        logger.info(f"[状态] 多头: ${status['total_long']:.0f} | "
                   f"空头: ${status['total_short']:.0f} | "
                   f"净Delta: ${status['net_delta']:.0f} | "
                   f"目标: ${status['target_exposure']:.0f} | "
                   f"偏差: {status['imbalance_pct']:.1%}")

        # 打印各仓位详情
        for pos in status['positions']:
            logger.info(f"  {pos['symbol']}: {pos['side']} "
                       f"${pos['current_notional']:.0f} / ${pos['target_notional']:.0f} "
                       f"({pos['fill_pct']:.0f}%) | 波动率: {pos['volatility']*100:.2f}%")

        # 检查是否需要调仓
        should, reason = self.strategy.should_rebalance()
        logger.info(f"[调仓检查] {reason}")

        if should:
            actions = self.strategy.calculate_rebalance_actions()
            if actions:
                logger.info(f"[调仓] 执行 {len(actions)} 个操作:")
                for action in actions:
                    success = self.execute_action(action)
                    if success and self.dry_run:
                        # 模拟更新持仓
                        pos = self.strategy.positions[action.symbol]
                        price = self.strategy.prices.get(action.symbol, 0)
                        if price > 0:
                            if pos.side == PositionSide.LONG:
                                if action.side == "BUY":
                                    new_qty = pos.quantity + action.quantity
                                else:
                                    new_qty = max(0, pos.quantity - action.quantity)
                            else:
                                if action.side == "SELL":
                                    new_qty = pos.quantity + action.quantity
                                else:
                                    new_qty = max(0, pos.quantity - action.quantity)
                            self.strategy.update_position(action.symbol, new_qty, price)

                self.strategy.record_rebalance(actions)

    def run(self):
        """主循环"""
        self.running = True
        self.start_time = time.time()

        logger.info("="*60)
        logger.info("波动率自适应交易器启动")
        logger.info("="*60)
        logger.info(f"模式: {'模拟' if self.dry_run else '实盘'}")
        logger.info(f"订单类型: {'限价单' if self.use_limit_orders else '市价单'}")
        logger.info(f"仓位范围: ${self.strategy.config.min_notional:,.0f} ~ ${self.strategy.config.max_notional:,.0f}")
        logger.info(f"波动率阈值: {self.strategy.config.volatility.vol_low*100:.1f}% ~ {self.strategy.config.volatility.vol_high*100:.1f}%")
        logger.info(f"挂单检查: 每{self.check_interval}秒")
        logger.info(f"调仓间隔: 每{self.rebalance_interval}秒 ({self.rebalance_interval//60}分钟)")
        logger.info("="*60)

        # 初始化波动率
        logger.info("获取初始波动率数据...")
        self.strategy.update_volatility(force=True)

        try:
            while self.running:
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"运行错误: {e}")

                runtime = (time.time() - self.start_time) / 60
                logger.info(f"[心跳] 运行 {runtime:.1f}分钟 | "
                           f"调仓 {self.strategy.rebalance_count}次 | "
                           f"交易量 ${self.strategy.total_volume:.0f}")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("收到停止信号")
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='波动率自适应交易器')
    parser.add_argument('--dry-run', action='store_true', help='模拟模式')
    parser.add_argument('--interval', type=int, default=60, help='挂单检查间隔秒数')
    parser.add_argument('--rebalance-interval', type=int, default=300, help='调仓间隔秒数(默认5分钟)')
    parser.add_argument('--limit-order', action='store_true', help='使用限价单')
    parser.add_argument('--offset', type=float, default=0.05, help='限价单偏移%')
    parser.add_argument('--min-notional', type=float, default=25000, help='最小单边仓位')
    parser.add_argument('--max-notional', type=float, default=75000, help='最大单边仓位')
    parser.add_argument('--vol-low', type=float, default=1.0, help='低波动阈值% (4h)')
    parser.add_argument('--vol-high', type=float, default=4.0, help='高波动阈值% (4h)')
    parser.add_argument('--yes', '-y', action='store_true', help='跳过主网确认')
    args = parser.parse_args()

    if not SDK_AVAILABLE:
        raise RuntimeError("apexomni SDK未安装")

    API_KEY = os.getenv("APEX_API_KEY", "")
    SECRET_KEY = os.getenv("APEX_SECRET_KEY", "")
    PASSPHRASE = os.getenv("APEX_PASSPHRASE", "")
    OMNIKEY = os.getenv("APEX_OMNIKEY", "")
    TESTNET = os.getenv("APEX_TESTNET", "true").lower() == "true"

    # 策略配置
    config = AdaptiveHedgingConfig(
        min_notional=args.min_notional,
        max_notional=args.max_notional,
        long_symbol="BTC-USDT",
        short_symbols=["ETH-USDT", "SOL-USDT"],
        volatility=VolatilityConfig(
            vol_low=args.vol_low / 100,
            vol_high=args.vol_high / 100
        ),
        leverage=3,
        rebalance_interval_seconds=60,
        scale_amount=2000.0,
        min_trade_notional=100.0,
        delta_threshold_pct=0.03,
        volatility_check_interval=300
    )

    print("="*60)
    print("波动率自适应Delta中性策略")
    print("="*60)
    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print(f"模式: {'模拟' if args.dry_run else '实盘'}")
    print(f"\n仓位范围:")
    print(f"  最小: ${config.min_notional:,.0f} (单边)")
    print(f"  最大: ${config.max_notional:,.0f} (单边)")
    print(f"\n波动率阈值:")
    print(f"  低波动 (<{args.vol_low}%): 满仓 ${config.max_notional*2:,.0f}")
    print(f"  高波动 (>{args.vol_high}%): 低仓 ${config.min_notional*2:,.0f}")
    print(f"\n币种配置:")
    print(f"  多头: {config.long_symbol}")
    print(f"  空头: {', '.join(config.short_symbols)} (按波动率反比分配)")
    print("="*60)

    if not TESTNET and not args.dry_run and not args.yes:
        confirm = input("确认主网实盘? (输入 YES): ")
        if confirm != "YES":
            print("已取消")
            return

    strategy = VolatilityAdaptiveStrategy(config)
    trader = ApexTrader(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        passphrase=PASSPHRASE,
        omnikey=OMNIKEY,
        testnet=TESTNET
    )

    account = trader.get_account()
    if account.get("success"):
        logger.info(f"账户连接成功")
    else:
        logger.error(f"账户连接失败: {account.get('error')}")
        if not args.dry_run:
            return

    adaptive_trader = VolatilityAdaptiveTrader(
        strategy=strategy,
        trader=trader,
        dry_run=args.dry_run,
        check_interval=args.interval,
        rebalance_interval=args.rebalance_interval,
        use_limit_orders=args.limit_order,
        price_offset_pct=args.offset / 100,
        testnet=TESTNET
    )

    adaptive_trader.run()


if __name__ == "__main__":
    main()
