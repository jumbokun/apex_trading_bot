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

    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "MARKET", reduce_only: bool = False) -> dict:
        """下单"""
        if not self.sign_client:
            logger.error("OMNIKEY未配置，无法下单")
            return {"success": False, "error": "OMNIKEY未配置"}

        try:
            size = self._round_size_to_step(size, symbol)

            order_params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type,
                "size": str(size),
                "reduceOnly": reduce_only,
            }

            result = self.sign_client.create_order_v3(**order_params)
            logger.info(f"下单成功: {result.get('data', {}).get('id', 'unknown')}")
            return {"success": True, "data": result.get("data", {})}

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {"success": False, "error": str(e)}


class HedgingTrader:
    """中性对冲交易器"""

    def __init__(self, strategy: NeutralHedgingStrategy, trader: ApexTrader,
                 dry_run: bool = True, check_interval: int = 60):
        self.strategy = strategy
        self.trader = trader
        self.dry_run = dry_run
        self.check_interval = check_interval
        self.running = False

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

        result = self.trader.place_order(
            symbol=action.symbol,
            side=action.side,
            size=action.quantity,
            order_type="MARKET"
        )

        if result.get("success"):
            logger.info(f"[实盘] 执行成功: {action}")
            self.trades_executed += 1
            self.total_volume += action.notional
            return True
        else:
            logger.error(f"[实盘] 执行失败: {action}, 错误: {result.get('error')}")
            return False

    def run_once(self):
        """运行一次检查和调仓"""
        # 1. 更新价格
        self.update_prices()

        # 2. 同步持仓（非模拟模式）
        if not self.dry_run:
            self.sync_positions()

        # 3. 打印当前状态
        summary = self.strategy.get_portfolio_summary()
        logger.info(f"[状态] 多头: ${summary['total_long']:.0f} | "
                   f"空头: ${summary['total_short']:.0f} | "
                   f"净Delta: ${summary['net_delta']:.0f} | "
                   f"偏差: {summary['imbalance_pct']:.1%} | "
                   f"中性: {'是' if summary['is_neutral'] else '否'}")

        # 4. 检查是否需要调仓
        should, reason = self.strategy.should_rebalance()
        logger.info(f"[调仓检查] {reason}")

        if should:
            # 5. 计算调仓操作
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
                            if action.side == "BUY":
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
        logger.info(f"检查间隔: {self.check_interval}秒")
        logger.info(f"币种配置:")
        for asset in self.strategy.config.assets:
            logger.info(f"  {asset.symbol}: {asset.side.value} ${asset.target_notional:.0f}")
        logger.info("="*60)

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
    if not SDK_AVAILABLE:
        raise RuntimeError("apexomni SDK未安装，请运行: pip install apexomni")

    # 从环境变量读取配置
    API_KEY = os.getenv("APEX_API_KEY", "")
    SECRET_KEY = os.getenv("APEX_SECRET_KEY", "")
    PASSPHRASE = os.getenv("APEX_PASSPHRASE", "")
    OMNIKEY = os.getenv("APEX_OMNIKEY", "")
    TESTNET = os.getenv("APEX_TESTNET", "true").lower() == "true"

    # 策略配置
    # 默认: BTC多$2000 + ETH空$4000 + SOL多$2000 = Delta中性
    config = HedgingConfig(
        assets=[
            AssetConfig("BTC-USDT", PositionSide.LONG, 2000.0),
            AssetConfig("ETH-USDT", PositionSide.SHORT, 4000.0),
            AssetConfig("SOL-USDT", PositionSide.LONG, 2000.0),
        ],
        leverage=3,
        rebalance_interval_seconds=900,  # 15分钟
        delta_threshold_pct=0.05,  # 5%偏差触发
        min_rebalance_interval_seconds=300,  # 最小5分钟间隔
        max_rebalances_per_hour=12,
        rebalance_step_pct=0.10,  # 每次调整10%
        min_trade_notional=50.0,
    )

    # 模拟/实盘模式
    DRY_RUN = True  # 改为False进行实盘交易

    print("="*60)
    print("中性对冲策略交易器")
    print("="*60)
    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print(f"模式: {'模拟运行' if DRY_RUN else '实盘交易'}")
    print(f"配置:")
    for asset in config.assets:
        print(f"  {asset.symbol}: {asset.side.value} ${asset.target_notional:.0f}")
    print(f"目标: 净Delta ≈ $0 (多空平衡)")
    print(f"调仓间隔: {config.rebalance_interval_seconds/60:.0f}分钟")
    print(f"偏差阈值: {config.delta_threshold_pct:.0%}")
    print("="*60)

    if not TESTNET and not DRY_RUN:
        print("\n⚠️  警告: 主网实盘交易模式!")
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
        check_interval=60  # 每分钟检查
    )

    hedging_trader.run()


if __name__ == "__main__":
    main()
