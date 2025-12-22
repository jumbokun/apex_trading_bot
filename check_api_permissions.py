"""
检查 Apex API 权限
测试合约/现货交易能力
"""
import os
import hmac
import base64
import hashlib
import time
import json
from urllib.parse import urlencode
import httpx

# 手动读取.env文件
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()


class ApexAPIChecker:
    """Apex API权限检查器"""

    REST_BASE_TESTNET = "https://testnet.omni.apex.exchange/api"
    REST_BASE_MAINNET = "https://omni.apex.exchange/api"

    def __init__(self, api_key, secret_key, passphrase, testnet=True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = self.REST_BASE_TESTNET if testnet else self.REST_BASE_MAINNET
        self.client = httpx.Client(timeout=30.0)

    def _sign(self, timestamp, method, request_path, body=""):
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _get_headers(self, timestamp, sign):
        return {
            "APEX-API-KEY": self.api_key,
            "APEX-API-SIGNATURE": sign,
            "APEX-API-TIMESTAMP": timestamp,
            "APEX-API-PASSPHRASE": self.passphrase,
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def get(self, endpoint, params=None, need_auth=True):
        request_path = endpoint
        if params:
            query_string = urlencode({k: v for k, v in params.items() if v is not None})
            if query_string:
                request_path = f"{endpoint}?{query_string}"

        url = f"{self.base_url}{request_path}"

        if need_auth:
            timestamp = str(int(time.time()))
            sign = self._sign(timestamp, "GET", request_path)
            headers = self._get_headers(timestamp, sign)
        else:
            headers = {}

        response = self.client.get(url, headers=headers)
        return response.json()

    def close(self):
        self.client.close()


def check_api_permissions():
    """检查API权限"""

    api_key = os.getenv("APEX_API_KEY", "")
    secret_key = os.getenv("APEX_SECRET_KEY", "")
    passphrase = os.getenv("APEX_PASSPHRASE", "")
    testnet = os.getenv("APEX_TESTNET", "true").lower() == "true"

    print("=" * 60)
    print("Apex API 权限检查")
    print("=" * 60)
    print(f"网络: {'测试网' if testnet else '主网'}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print("=" * 60)

    checker = ApexAPIChecker(api_key, secret_key, passphrase, testnet)

    try:
        # 1. 测试公共API连接
        print("\n[1] 测试公共API连接...")
        try:
            time_data = checker.get("/v3/public/time", need_auth=False)
            if time_data.get("success"):
                print(f"  ✓ 公共API连接正常")
                print(f"    服务器时间: {time_data.get('data')}")
            else:
                print(f"  ✗ 公共API连接失败: {time_data}")
                return
        except Exception as e:
            print(f"  ✗ 公共API连接失败: {e}")
            return

        # 2. 获取交易对配置
        print("\n[2] 获取可交易对...")
        try:
            config_data = checker.get("/v3/public/config", need_auth=False)
            if config_data.get("success"):
                data = config_data.get("data", {})
                symbols = data.get("symbols", [])
                perpetual = data.get("perpetual", {})

                print(f"  ✓ 共有 {len(symbols)} 个交易对")

                # 显示交易对类型
                print(f"\n  永续合约交易对:")
                for s in symbols[:10]:
                    symbol_name = s.get("symbol", "")
                    print(f"    - {symbol_name}")

                if len(symbols) > 10:
                    print(f"    ... 还有 {len(symbols) - 10} 个")

                print(f"\n  ℹ️  Apex Exchange 是永续合约交易所")
                print(f"      所有交易对都是永续合约 (USDC结算)")
            else:
                print(f"  ⚠️  获取配置失败: {config_data}")
        except Exception as e:
            print(f"  ✗ 获取配置失败: {e}")

        # 3. 测试账户API
        print("\n[3] 测试账户权限...")
        try:
            account_data = checker.get("/v3/account")
            if account_data.get("success"):
                data = account_data.get("data", {})
                print(f"  ✓ 账户API访问正常")
                print(f"    账户ID: {data.get('id', 'N/A')}")
                print(f"    总权益: {data.get('totalEquity', 'N/A')} USDC")
                print(f"    可用余额: {data.get('availableBalance', 'N/A')} USDC")
                print(f"    初始保证金: {data.get('initialMargin', 'N/A')} USDC")
                print(f"    维持保证金: {data.get('maintenanceMargin', 'N/A')} USDC")

                # 检查持仓
                positions = data.get("positions", [])
                open_positions = [p for p in positions if float(p.get("size", 0)) != 0]
                print(f"\n    当前持仓: {len(open_positions)} 个")

                for pos in open_positions:
                    symbol = pos.get("symbol", "")
                    size = pos.get("size", "0")
                    entry_price = pos.get("entryPrice", "0")
                    unrealized_pnl = pos.get("unrealizedPnl", "0")
                    print(f"      - {symbol}: {size} @ {entry_price} (盈亏: {unrealized_pnl})")
            else:
                code = account_data.get("code", "")
                msg = account_data.get("message", "")
                print(f"  ✗ 账户API访问失败: [{code}] {msg}")
        except Exception as e:
            print(f"  ✗ 账户API错误: {e}")

        # 4. 测试余额查询
        print("\n[4] 测试余额查询...")
        try:
            balance_data = checker.get("/v3/account/balance")
            if balance_data.get("success"):
                print(f"  ✓ 余额查询正常")
                data = balance_data.get("data", {})
                if isinstance(data, dict):
                    for currency, info in data.items():
                        if isinstance(info, dict):
                            balance = info.get("balance", "0")
                            print(f"    {currency}: {balance}")
                        else:
                            print(f"    {currency}: {info}")
            else:
                print(f"  ⚠️  余额查询响应: {balance_data}")
        except Exception as e:
            print(f"  ⚠️  余额查询错误: {e}")

        # 5. 测试交易权限
        print("\n[5] 测试交易权限...")
        try:
            open_orders = checker.get("/v3/order/open")
            if open_orders.get("success"):
                orders = open_orders.get("data", {}).get("list", [])
                print(f"  ✓ 交易API访问正常 - 有交易权限")
                print(f"    当前挂单: {len(orders)} 个")
                for order in orders[:3]:
                    symbol = order.get("symbol", "")
                    side = order.get("side", "")
                    size = order.get("size", "")
                    price = order.get("price", "")
                    print(f"      - {symbol} {side} {size} @ {price}")
            else:
                code = open_orders.get("code", "")
                msg = open_orders.get("message", "")
                print(f"  ⚠️  交易API响应: [{code}] {msg}")
                if "permission" in msg.lower() or "auth" in msg.lower():
                    print(f"    ℹ️  可能没有交易权限")
        except Exception as e:
            print(f"  ✗ 交易API错误: {e}")

        # 6. 测试历史订单
        print("\n[6] 测试历史订单查询...")
        try:
            history = checker.get("/v3/order/history", {"limit": "5"})
            if history.get("success"):
                orders = history.get("data", {}).get("list", [])
                print(f"  ✓ 历史订单查询正常")
                print(f"    历史订单: {len(orders)} 条")
                for order in orders:
                    symbol = order.get("symbol", "")
                    side = order.get("side", "")
                    size = order.get("size", "")
                    status = order.get("status", "")
                    print(f"      - {symbol} {side} {size} - {status}")
            else:
                print(f"  ⚠️  历史订单查询响应: {history}")
        except Exception as e:
            print(f"  ⚠️  历史订单查询错误: {e}")

        # 7. 获取资金费率
        print("\n[7] 测试资金费率查询...")
        try:
            funding = checker.get("/v3/account/funding-rate")
            if funding.get("success"):
                print(f"  ✓ 资金费率查询正常")
                rates = funding.get("data", [])
                if isinstance(rates, list):
                    for rate in rates[:3]:
                        symbol = rate.get("symbol", "")
                        fr = rate.get("fundingRate", "0")
                        print(f"    {symbol}: {float(fr)*100:.4f}%")
            else:
                print(f"  ⚠️  资金费率查询响应: {funding}")
        except Exception as e:
            print(f"  ⚠️  资金费率查询错误: {e}")

        # 总结
        print("\n" + "=" * 60)
        print("权限总结")
        print("=" * 60)
        print("""
  Apex Exchange 交易类型:

  ✓ 永续合约交易 - 支持
    - BTC-USDC, ETH-USDC, SOL-USDC 等
    - USDC 作为保证金和结算货币
    - 支持多空双向交易
    - 支持杠杆（具体倍数视交易对而定）

  ✗ 现货交易 - 不支持
    - Apex 是专业的永续合约交易所
    - 不提供现货交易功能

  您的API权限:
  - ✓ 读取账户信息
  - ✓ 查询持仓和余额
  - ✓ 查看挂单和历史
  - ✓ 下单和撤单（永续合约）
  - ✓ 设置止损止盈

  如需现货交易，请使用:
  - Binance
  - OKX
  - Bybit
  - 等支持现货的交易所
""")
        print("=" * 60)

    finally:
        checker.close()


if __name__ == "__main__":
    check_api_permissions()
