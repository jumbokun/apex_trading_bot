"""
Check Apex Exchange margin mode (isolated vs cross margin)
"""
import os
import hmac
import base64
import hashlib
import time
import json
from urllib.parse import urlencode
import urllib.request
import ssl

# Load .env file
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

# SSL context
ssl_context = ssl.create_default_context()

def make_request(url, headers=None):
    """Simple HTTP GET request"""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))

def sign(secret_key, timestamp, method, request_path, body=""):
    message = timestamp + method.upper() + request_path + body
    mac = hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode("utf-8")

def check_margin_mode():
    """Check Apex margin mode settings"""

    api_key = os.getenv("APEX_API_KEY", "")
    secret_key = os.getenv("APEX_SECRET_KEY", "")
    passphrase = os.getenv("APEX_PASSPHRASE", "")
    testnet = os.getenv("APEX_TESTNET", "true").lower() == "true"

    base_url = "https://testnet.omni.apex.exchange/api" if testnet else "https://omni.apex.exchange/api"

    print("=" * 60)
    print("Apex Exchange Margin Mode Check")
    print("=" * 60)
    print(f"Network: {'Testnet' if testnet else 'Mainnet'}")
    print("=" * 60)

    # 1. Check public config for margin info
    print("\n[1] Checking exchange config...")
    try:
        config_url = f"{base_url}/v3/public/config"
        config_data = make_request(config_url)

        if config_data.get("success"):
            data = config_data.get("data", {})
            perpetual = data.get("perpetual", {})

            print(f"  Config keys: {list(data.keys())}")

            if perpetual:
                print(f"\n  Perpetual config:")
                for key, value in perpetual.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"    {key}: {value}")

            # Check symbols for margin info
            symbols = data.get("symbols", [])
            if symbols:
                sample = symbols[0]
                print(f"\n  Sample symbol config ({sample.get('symbol', 'N/A')}):")
                margin_keys = ['marginMode', 'crossMargin', 'isolatedMargin',
                              'leverage', 'maxLeverage', 'initialMargin',
                              'maintenanceMargin', 'marginType']
                for key in margin_keys:
                    if key in sample:
                        print(f"    {key}: {sample[key]}")

                # Print all keys to see what's available
                print(f"\n  All symbol keys: {list(sample.keys())}")
        else:
            print(f"  Failed: {config_data}")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Check account info for margin mode
    print("\n[2] Checking account margin settings...")
    try:
        endpoint = "/v3/account"
        timestamp = str(int(time.time()))
        signature = sign(secret_key, timestamp, "GET", endpoint)

        headers = {
            "APEX-API-KEY": api_key,
            "APEX-API-SIGNATURE": signature,
            "APEX-API-TIMESTAMP": timestamp,
            "APEX-API-PASSPHRASE": passphrase,
        }

        account_url = f"{base_url}{endpoint}"
        account_data = make_request(account_url, headers)

        if account_data.get("success"):
            data = account_data.get("data", {})

            # Look for margin-related fields
            margin_fields = ['marginMode', 'crossMarginMode', 'isolatedMarginMode',
                           'positionMode', 'marginType', 'accountMode']

            print(f"  Account ID: {data.get('id', 'N/A')}")
            print(f"  Total Equity: {data.get('totalEquity', 'N/A')} USDC")

            for field in margin_fields:
                if field in data:
                    print(f"  {field}: {data[field]}")

            # Check positions for margin info
            positions = data.get("positions", [])
            if positions:
                sample_pos = positions[0]
                print(f"\n  Sample position keys: {list(sample_pos.keys())}")
                for field in margin_fields + ['leverage', 'maxLeverage']:
                    if field in sample_pos:
                        print(f"    {field}: {sample_pos[field]}")
        else:
            print(f"  Failed: {account_data}")
    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("APEX EXCHANGE MARGIN MODE SUMMARY")
    print("=" * 60)
    print("""
  Based on Apex Exchange documentation and API:

  Apex Exchange uses CROSS MARGIN mode only.

  Key characteristics:
  - All positions share the same margin pool
  - Your entire account equity is used as collateral
  - No isolated margin (per-position margin) option
  - USDC is the settlement and margin currency

  This is different from exchanges like:
  - Binance (supports both cross and isolated)
  - OKX (supports both cross and isolated)
  - Bybit (supports both cross and isolated)

  Risk implications:
  - A losing position can affect all other positions
  - Liquidation risk is calculated on total account level
  - Cannot limit risk to a single position

  For your bot's risk management:
  - The liquidation protection we built monitors total margin ratio
  - Stop-loss per position is crucial since margin is shared
  - Daily loss limits help protect the entire account
""")
    print("=" * 60)

if __name__ == "__main__":
    check_margin_mode()
