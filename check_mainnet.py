"""检查主网连接"""
import os
import json
import time
import hmac
import base64
import hashlib
import urllib.request
import ssl

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
print("Apex Exchange 连接检查")
print("=" * 50)
print(f"网络: {'测试网' if TESTNET else '主网'}")
print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
print(f"Base URL: {BASE_URL}")
print("=" * 50)

ssl_ctx = ssl.create_default_context()

def sign(timestamp, method, path, body=""):
    # Apex使用秒级时间戳
    message = timestamp + method.upper() + path + body
    mac = hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode('utf-8')

def get_timestamp():
    # 使用秒级时间戳
    return str(int(time.time()))

def request(endpoint):
    url = BASE_URL + endpoint
    timestamp = get_timestamp()  # 使用秒级时间戳
    signature = sign(timestamp, "GET", endpoint)

    headers = {
        "APEX-API-KEY": API_KEY,
        "APEX-API-SIGNATURE": signature,
        "APEX-API-TIMESTAMP": timestamp,
        "APEX-API-PASSPHRASE": PASSPHRASE,
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"success": False, "error": str(e)}

# 测试公共API
print("\n[1] 测试公共API...")
try:
    # 尝试获取ticker数据
    url = BASE_URL + "/v3/public/ticker?symbol=BTC-USDC"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        if data.get("success"):
            print("  [OK] 公共API正常")
            tickers = data.get("data", [])
            if tickers:
                btc = tickers[0]
                print(f"    BTC-USDC: {btc.get('lastPrice', 'N/A')}")
        else:
            print(f"  [FAIL] 公共API失败: {data}")
except Exception as e:
    print(f"  [FAIL] 公共API错误: {e}")

# 测试账户API
print("\n[2] 测试账户API...")
result = request("/v3/account")
if result.get("success"):
    data = result.get("data", {})
    print("  [OK] 账户API正常")
    print(f"    账户ID: {data.get('id', 'N/A')}")
    print(f"    总权益: {data.get('totalEquity', '0')} USDC")
    print(f"    可用余额: {data.get('availableBalance', '0')} USDC")

    # 检查持仓
    positions = data.get("positions", [])
    open_pos = [p for p in positions if float(p.get("size", 0)) != 0]
    print(f"    当前持仓: {len(open_pos)} 个")
    for p in open_pos:
        print(f"      - {p.get('symbol')}: {p.get('size')} @ {p.get('entryPrice')}")
else:
    print(f"  [FAIL] 账户API失败: {result}")
    if "error" in result:
        print(f"    错误详情: {result['error']}")

print("\n" + "=" * 50)
