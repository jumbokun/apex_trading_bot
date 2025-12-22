"""使用官方apexomni SDK测试连接"""
import os

# 加载.env
env_path = os.path.join(os.path.dirname(__file__), '.env')
with open(env_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

API_KEY = os.getenv('APEX_API_KEY', '')
SECRET_KEY = os.getenv('APEX_SECRET_KEY', '')
PASSPHRASE = os.getenv('APEX_PASSPHRASE', '')
TESTNET = os.getenv('APEX_TESTNET', 'true').lower() == 'true'

# Apex Omni network IDs
NETWORK_ID_MAINNET = 42161  # Arbitrum mainnet
NETWORK_ID_TESTNET = 421614  # Arbitrum Sepolia testnet

print("=" * 50)
print("Apex Exchange Official SDK Test")
print("=" * 50)
print(f"Network: {'Testnet' if TESTNET else 'MAINNET'}")
print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
print("=" * 50)

from apexomni.http_public import HttpPublic
from apexomni.http_private import HttpPrivate

# Set endpoint based on network
if TESTNET:
    endpoint = "https://testnet.omni.apex.exchange"
    network_id = NETWORK_ID_TESTNET
else:
    endpoint = "https://omni.apex.exchange"
    network_id = NETWORK_ID_MAINNET

print(f"\nEndpoint: {endpoint}")
print(f"Network ID: {network_id}")

# Test public API
print("\n[1] Testing Public API...")
try:
    public_client = HttpPublic(endpoint)

    # Get server time
    server_time = public_client.server_time()
    print(f"  [OK] Server Time: {server_time}")

    # Get ticker
    ticker = public_client.ticker_v3(symbol="BTC-USDC")
    if ticker.get("data"):
        btc_data = ticker["data"][0] if isinstance(ticker["data"], list) else ticker["data"]
        print(f"  [OK] BTC-USDC: ${btc_data.get('lastPrice', 'N/A')}")
    else:
        print(f"  [INFO] Ticker response: {ticker}")
except Exception as e:
    print(f"  [FAIL] Public API error: {e}")

# Test private API
print("\n[2] Testing Private API (Account)...")
try:
    private_client = HttpPrivate(
        endpoint,
        network_id=network_id,
        api_key_credentials={
            'key': API_KEY,
            'secret': SECRET_KEY,
            'passphrase': PASSPHRASE
        }
    )

    # Get account (使用get_account_v2)
    account = private_client.get_account_v2()
    print(f"  [INFO] Raw response: {account}")

    # 检查是否有data字段
    if account.get("data"):
        data = account.get("data", {})
        print("  [OK] Account API working")
        print(f"    Account ID: {data.get('id', 'N/A')}")
        print(f"    Stark Key: {data.get('starkKey', 'N/A')[:20] + '...' if data.get('starkKey') else 'N/A'}")
        print(f"    Ethereum Address: {data.get('ethereumAddress', 'N/A')}")

        # 获取账户余额 - 使用get_account_balance
        try:
            balance = private_client.get_account_balance()
            print(f"\n  [INFO] Balance response: {balance}")
            if balance.get("data"):
                bal_data = balance.get("data", {})
                print(f"    Total Equity: {bal_data.get('totalEquity', '0')} USDC")
                print(f"    Available Balance: {bal_data.get('availableBalance', '0')} USDC")
        except Exception as e:
            print(f"  [WARN] Could not get balance v1: {e}")
            # 尝试直接获取账户
            try:
                acc = private_client.account()
                print(f"\n  [INFO] Account response: {acc}")
            except Exception as e2:
                print(f"  [WARN] Could not get account: {e2}")
    else:
        print(f"  [FAIL] Account API error: {account}")
except Exception as e:
    print(f"  [FAIL] Private API error: {e}")

print("\n" + "=" * 50)
print("Test complete!")
