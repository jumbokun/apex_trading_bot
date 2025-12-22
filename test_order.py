"""
快速测试下单模块 - 开一个小空单然后立刻平掉
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# 加载环境变量
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

# 导入SDK
try:
    from apexomni.http_public import HttpPublic
    from apexomni.http_private import HttpPrivate
    from apexomni.http_private_sign import HttpPrivateSign
except ImportError:
    print("请安装 apexomni: pip install apexomni")
    sys.exit(1)

APEX_ENDPOINT = "https://omni.apex.exchange"
NETWORK_ID = 42161  # Arbitrum mainnet

def main():
    print("=" * 50)
    print("下单模块测试 - BTC LONG 最小单 (合约)")
    print("=" * 50)

    # 获取配置
    api_key = os.environ.get("APEX_API_KEY", "")
    secret_key = os.environ.get("APEX_SECRET_KEY", "")
    passphrase = os.environ.get("APEX_PASSPHRASE", "")
    omnikey = os.environ.get("OMNIKEY", "")

    if not all([api_key, secret_key, passphrase, omnikey]):
        print("错误: API配置不完整")
        return

    # 初始化客户端 (与live_trader_v2相同方式)
    public_client = HttpPublic(APEX_ENDPOINT)

    sign_client = HttpPrivateSign(
        APEX_ENDPOINT,
        network_id=NETWORK_ID,
        zk_seeds=omnikey,
        zk_l2Key=omnikey,
        api_key_credentials={
            'key': api_key,
            'secret': secret_key,
            'passphrase': passphrase
        }
    )

    print("[OK] 客户端初始化成功")

    # 先调用get_account_v3来初始化accountV3
    try:
        account = sign_client.get_account_v3()
        print(f"[OK] 账户初始化成功: {account.get('id', 'N/A')}")
    except Exception as e:
        print(f"[FAIL] 账户初始化失败: {e}")
        return

    # 调用configs_v3来初始化configV3
    try:
        configs = sign_client.configs_v3()
        print(f"[OK] 配置初始化成功")
    except Exception as e:
        print(f"[FAIL] 配置初始化失败: {e}")
        return

    # 获取BTC当前价格
    symbol = "BTC-USDT"
    ticker = public_client.ticker_v3(symbol=symbol)
    print(f"Ticker响应: {ticker}")

    if not ticker.get("data"):
        print(f"错误: 无法获取{symbol}价格")
        return

    ticker_data = ticker["data"]
    # 可能是list或dict
    if isinstance(ticker_data, list):
        # 找到对应symbol
        for t in ticker_data:
            if t.get("symbol") == symbol:
                current_price = float(t.get("lastPrice", 0))
                break
        else:
            current_price = float(ticker_data[0].get("lastPrice", 0)) if ticker_data else 0
    else:
        current_price = float(ticker_data.get("lastPrice", 0))

    print(f"[OK] {symbol} 当前价格: {current_price:.2f}")

    # 计算最小下单量 (约10 USDT名义价值)
    min_size = 0.001  # BTC最小下单量
    notional = min_size * current_price
    print(f"[OK] 下单数量: {min_size} BTC (约 {notional:.2f} USDT)")

    # 开多单 (合约)
    print("\n--- 开多单 (永续合约) ---")
    order_params = {
        "symbol": symbol,
        "side": "BUY",  # 多单用BUY开仓
        "type": "MARKET",
        "size": str(min_size),
        "price": str(current_price),  # SDK需要price参数
        "timestampSeconds": time.time(),
        "reduceOnly": False
    }

    print(f"订单参数: {order_params}")

    try:
        result = sign_client.create_order_v3(**order_params)
        print(f"开多结果: {result}")

        if result.get("data"):
            order_id = result["data"].get("orderId")
            print(f"[OK] 开多成功! 订单ID: {order_id}")
        else:
            print(f"[FAIL] 开多失败: {result}")
            return

    except Exception as e:
        print(f"[FAIL] 开多异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # 等待2秒让订单成交
    print("\n等待2秒...")
    time.sleep(2)

    # 平仓
    print("\n--- 平多仓 ---")
    close_params = {
        "symbol": symbol,
        "side": "SELL",  # 多单用SELL平仓
        "type": "MARKET",
        "size": str(min_size),
        "price": str(current_price),
        "timestampSeconds": time.time(),
        "reduceOnly": True
    }

    print(f"平仓参数: {close_params}")

    try:
        result = sign_client.create_order_v3(**close_params)
        print(f"平仓结果: {result}")

        if result.get("data"):
            order_id = result["data"].get("orderId")
            print(f"[OK] 平多成功! 订单ID: {order_id}")
        else:
            print(f"[FAIL] 平多失败: {result}")

    except Exception as e:
        print(f"[FAIL] 平仓异常: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
