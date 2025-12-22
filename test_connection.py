"""测试主网连接 - 独立脚本"""
import os
import sys
import hmac
import base64
import hashlib
import time
import json
import httpx

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

BASE_URL = "https://testnet.omni.apex.exchange/api" if TESTNET else "https://omni.apex.exchange/api"

print("=" * 50)
print("Apex Exchange Connection Test")
print("=" * 50)
print(f"Network: {'Testnet' if TESTNET else 'MAINNET'}")
print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
print(f"Base URL: {BASE_URL}")
print("=" * 50)

def get_timestamp():
    return str(int(time.time()))

def sign(timestamp, method, path, body=""):
    message = timestamp + method.upper() + path + body
    mac = hmac.new(
        SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode('utf-8')

def request(method, endpoint, params=None, body=None, need_auth=True):
    from urllib.parse import urlencode

    request_path = endpoint
    if params:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            request_path = f"{endpoint}?{query}"

    url = f"{BASE_URL}{request_path}"

    headers = {}
    if need_auth:
        timestamp = get_timestamp()
        body_str = ""
        signature = sign(timestamp, method, request_path, body_str)
        headers = {
            "APEX-API-KEY": API_KEY,
            "APEX-API-SIGNATURE": signature,
            "APEX-API-TIMESTAMP": timestamp,
            "APEX-API-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/x-www-form-urlencoded",
        }

    with httpx.Client(timeout=30.0) as client:
        if method.upper() == "GET":
            response = client.get(url, headers=headers)
        else:
            response = client.post(url, headers=headers, content=body_str if body else None)

        return response.json()

# Test public API
print("\n[1] Testing Public API...")
try:
    result = request("GET", "/v3/public/ticker", {"symbol": "BTC-USDC"}, need_auth=False)
    if result.get("success"):
        data = result.get("data", [])
        if data:
            btc = data[0]
            print(f"  [OK] Public API working")
            print(f"    BTC-USDC: ${btc.get('lastPrice', 'N/A')}")
    else:
        print(f"  [FAIL] {result}")
except Exception as e:
    print(f"  [FAIL] Public API error: {e}")

# Test account API
print("\n[2] Testing Account API...")
try:
    result = request("GET", "/v3/account")
    if result.get("success"):
        data = result.get("data", {})
        print("  [OK] Account API working")
        print(f"    Account ID: {data.get('id', 'N/A')}")
        print(f"    Total Equity: {data.get('totalEquity', '0')} USDC")
        print(f"    Available Balance: {data.get('availableBalance', '0')} USDC")

        # Check positions
        positions = data.get("positions", [])
        open_pos = [p for p in positions if float(p.get("size", 0)) != 0]
        print(f"    Open Positions: {len(open_pos)}")
        for p in open_pos:
            print(f"      - {p.get('symbol')}: {p.get('size')} @ {p.get('entryPrice')}")
    else:
        print(f"  [FAIL] API Error: {result}")
except Exception as e:
    print(f"  [FAIL] Account API error: {e}")

print("\n" + "=" * 50)
