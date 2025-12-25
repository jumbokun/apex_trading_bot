"""
获取Apex Exchange最近一周的交易量和手续费统计
使用OMNIKEY查询账户交易历史
"""
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

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
    from apexomni.http_private import HttpPrivate
    from apexomni.http_private_sign import HttpPrivateSign
except ImportError:
    print("请安装apexomni: pip install apexomni")
    sys.exit(1)

# 配置
API_KEY = os.getenv("APEX_API_KEY", "")
SECRET_KEY = os.getenv("APEX_SECRET_KEY", "")
PASSPHRASE = os.getenv("APEX_PASSPHRASE", "")
OMNIKEY = os.getenv("APEX_OMNIKEY", "")
TESTNET = os.getenv("APEX_TESTNET", "true").lower() == "true"

if TESTNET:
    ENDPOINT = "https://testnet.omni.apex.exchange"
    NETWORK_ID = 421614
else:
    ENDPOINT = "https://omni.apex.exchange"
    NETWORK_ID = 42161


def get_trade_stats():
    """获取交易统计"""
    print("=" * 60)
    print("Apex Exchange 交易统计")
    print("=" * 60)
    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print(f"Endpoint: {ENDPOINT}")
    print()

    # 初始化客户端
    if OMNIKEY:
        print("使用OMNIKEY初始化...")
        client = HttpPrivateSign(
            ENDPOINT,
            network_id=NETWORK_ID,
            zk_seeds=OMNIKEY,
            zk_l2Key=OMNIKEY,
            api_key_credentials={
                'key': API_KEY,
                'secret': SECRET_KEY,
                'passphrase': PASSPHRASE
            }
        )
    else:
        print("使用API Key初始化...")
        client = HttpPrivate(
            ENDPOINT,
            network_id=NETWORK_ID,
            api_key_credentials={
                'key': API_KEY,
                'secret': SECRET_KEY,
                'passphrase': PASSPHRASE
            }
        )

    # 计算时间范围 (最近7天)
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

    print(f"查询时间范围: {datetime.fromtimestamp(start_time/1000)} ~ {datetime.fromtimestamp(end_time/1000)}")
    print()

    # 先加载配置
    print("加载配置...")
    try:
        client.configs_v3()
        print("  配置加载成功")
    except Exception as e:
        print(f"  配置加载失败: {e}")

    # 获取成交记录 (fills) - 使用v3版本
    print("获取成交记录...")
    all_fills = []

    try:
        # 使用v3版本API - 注意返回的是orders字段
        result = client.fills_v3(
            beginTimeInclusive=start_time,
            endTimeExclusive=end_time,
            limit=100
        )

        # fills_v3返回的是orders字段
        orders = result.get("data", {}).get("orders", [])
        if orders:
            # 过滤只保留成功的订单
            for order in orders:
                if order.get("status") == "SUCCESS_L2_APPROVED":
                    all_fills.append(order)
            print(f"  获取到 {len(all_fills)} 条成交记录")

            # 如果有分页，继续获取
            while len(orders) >= 100:
                last_time = orders[-1].get("createdAt", 0)
                result = client.fills_v3(
                    beginTimeInclusive=start_time,
                    endTimeExclusive=last_time,
                    limit=100
                )
                orders = result.get("data", {}).get("orders", [])
                for order in orders:
                    if order.get("status") == "SUCCESS_L2_APPROVED":
                        all_fills.append(order)
                print(f"  继续获取，共 {len(all_fills)} 条")

    except Exception as e:
        print(f"  fills_v3失败: {e}")
        import traceback
        traceback.print_exc()

    if not all_fills:
        print("\n最近一周没有成交记录")

        # 尝试获取历史订单 - v3
        print("\n尝试获取历史订单...")
        try:
            result = client.history_orders_v3()
            print(f"  API响应: {result}")
            orders = result.get("data", {}).get("orders", [])
            print(f"历史订单数: {len(orders)}")
            if orders:
                print("\n最近5条订单:")
                for o in orders[:5]:
                    print(f"  {o.get('symbol')} {o.get('side')} {o.get('size')} @ {o.get('price')} - {o.get('status')}")
        except Exception as e:
            print(f"获取历史订单失败: {e}")

        # 尝试获取账户信息
        print("\n获取账户信息...")
        try:
            result = client.get_account_v3()
            print(f"账户信息: {result}")
        except Exception as e:
            print(f"获取账户信息失败: {e}")

        return

    # 统计
    print(f"\n共获取到 {len(all_fills)} 条成交记录")
    print()

    # 按币种统计
    stats_by_symbol = defaultdict(lambda: {"volume": 0, "fee": 0, "trades": 0, "buy_volume": 0, "sell_volume": 0})
    total_volume = 0
    total_fee = 0
    total_trades = len(all_fills)

    for fill in all_fills:
        symbol = fill.get("symbol", "UNKNOWN")
        size = float(fill.get("size", 0))
        price = float(fill.get("price", 0))
        fee = float(fill.get("fee", 0))
        side = fill.get("side", "")

        volume = size * price
        total_volume += volume
        total_fee += fee

        stats_by_symbol[symbol]["volume"] += volume
        stats_by_symbol[symbol]["fee"] += fee
        stats_by_symbol[symbol]["trades"] += 1
        if side == "BUY":
            stats_by_symbol[symbol]["buy_volume"] += volume
        else:
            stats_by_symbol[symbol]["sell_volume"] += volume

    # 输出统计结果
    print("=" * 60)
    print("最近一周交易统计汇总")
    print("=" * 60)
    print(f"总交易次数: {total_trades}")
    print(f"总交易量: ${total_volume:,.2f}")
    print(f"总手续费: ${total_fee:,.4f}")
    print()

    print("按币种统计:")
    print("-" * 60)
    print(f"{'币种':<16} {'交易次数':>10} {'交易量(USD)':>16} {'手续费(USD)':>14}")
    print("-" * 60)

    for symbol in sorted(stats_by_symbol.keys()):
        s = stats_by_symbol[symbol]
        print(f"{symbol:<16} {s['trades']:>10} {s['volume']:>16,.2f} {s['fee']:>14,.4f}")

    print("-" * 60)
    print(f"{'合计':<16} {total_trades:>10} {total_volume:>16,.2f} {total_fee:>14,.4f}")
    print("=" * 60)

    # 按日期统计
    print("\n按日期统计:")
    print("-" * 60)
    stats_by_date = defaultdict(lambda: {"volume": 0, "fee": 0, "trades": 0})

    for fill in all_fills:
        created_at = fill.get("createdAt", "")
        if isinstance(created_at, int):
            date_str = datetime.fromtimestamp(created_at / 1000).strftime("%Y-%m-%d")
        else:
            try:
                date_str = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%Y-%m-%d")
            except:
                date_str = "未知日期"

        size = float(fill.get("size", 0))
        price = float(fill.get("price", 0))
        fee = float(fill.get("fee", 0))

        stats_by_date[date_str]["volume"] += size * price
        stats_by_date[date_str]["fee"] += fee
        stats_by_date[date_str]["trades"] += 1

    print(f"{'日期':<12} {'交易次数':>10} {'交易量(USD)':>16} {'手续费(USD)':>14}")
    print("-" * 60)

    for date in sorted(stats_by_date.keys()):
        s = stats_by_date[date]
        print(f"{date:<12} {s['trades']:>10} {s['volume']:>16,.2f} {s['fee']:>14,.4f}")


if __name__ == "__main__":
    get_trade_stats()
