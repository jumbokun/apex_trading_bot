"""
查询 Apex Exchange 资金费率
使用公开API: /api/v3/ticker (实时费率)
"""
import os
import sys
import json
import urllib.request
import ssl
from datetime import datetime


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


def fetch_funding_rate(endpoint: str, symbol: str) -> dict:
    """获取实时资金费率 (通过ticker接口)"""
    ssl_ctx = ssl.create_default_context()
    # 转换 symbol 格式: BTC-USDT -> BTCUSDT
    ticker_symbol = symbol.replace("-", "")
    url = f"{endpoint}/api/v3/ticker?symbol={ticker_symbol}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"请求失败: {e}")
        return {}


def main():
    # 使用主网
    TESTNET = os.getenv("APEX_TESTNET", "false").lower() == "true"

    if TESTNET:
        endpoint = "https://testnet.omni.apex.exchange"
    else:
        endpoint = "https://omni.apex.exchange"

    print(f"网络: {'测试网' if TESTNET else '主网'}")
    print("=" * 60)

    # 查询所有合约的资金费率
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    print(f"{'币种':<12} {'当前费率':<15} {'建议'}")
    print("-" * 60)

    funding_data = []

    for symbol in symbols:
        try:
            # 获取实时资金费率
            result = fetch_funding_rate(endpoint, symbol)
            tickers = result.get("data", [])

            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                funding_rate = float(ticker.get("fundingRate", 0))
                predicted_rate = float(ticker.get("predictedFundingRate", 0))
                next_time = ticker.get("nextFundingTime", "")

                # 解析下次结算时间
                if next_time:
                    try:
                        dt = datetime.fromisoformat(next_time.replace("Z", "+00:00"))
                        time_str = dt.strftime("%H:%M")
                    except:
                        time_str = next_time
                else:
                    time_str = "N/A"

                funding_data.append({
                    "symbol": symbol,
                    "rate": funding_rate,
                    "predicted": predicted_rate,
                    "next_time": time_str
                })

                # 判断建议
                # 正费率: 多头付费给空头，做空有利
                # 负费率: 空头付费给多头，做多有利
                if funding_rate > 0.000001:  # 正费率
                    suggestion = "SHORT (多付空)"
                elif funding_rate < -0.000001:  # 负费率
                    suggestion = "LONG (空付多)"
                else:
                    suggestion = "中性"

                print(f"{symbol:<12} {funding_rate*100:>+.6f}%    {suggestion}")
            else:
                print(f"{symbol:<12} 无数据")

        except Exception as e:
            print(f"{symbol:<12} 获取失败: {e}")

    # 显示下次结算时间
    if funding_data and funding_data[0].get("next_time"):
        next_time_display = funding_data[0]["next_time"]
        # 计算距离结算时间
        try:
            result = fetch_funding_rate(endpoint, symbols[0])
            tickers = result.get("data", [])
            if tickers:
                next_time_str = tickers[0].get("nextFundingTime", "")
                if next_time_str:
                    from datetime import timezone
                    dt = datetime.fromisoformat(next_time_str.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    delta = dt - now
                    secs = max(0, int(delta.total_seconds()))
                    print(f"\n下次结算: {next_time_display} UTC (还有 {secs//60}分{secs%60}秒)")
                else:
                    print(f"\n下次结算: {next_time_display} UTC")
            else:
                print(f"\n下次结算: {next_time_display} UTC")
        except Exception as e:
            print(f"\n下次结算: {next_time_display} UTC")

    print("=" * 60)

    # 根据费率排序，推荐策略
    if funding_data:
        # 按费率排序 (高到低)
        funding_data.sort(key=lambda x: x["rate"], reverse=True)

        highest = funding_data[0]
        lowest = funding_data[-1]
        middle = funding_data[1] if len(funding_data) > 2 else None

        print("\n[费率排序] 高 -> 低:")
        for i, d in enumerate(funding_data, 1):
            print(f"  {i}. {d['symbol']}: {d['rate']*100:+.4f}%")

        # 计算中性配置
        print("\n[Delta中性配置建议]")
        print("  (正费率币种做空收费, 负费率币种做多收费)")

        if middle:
            # 根据费率决定方向
            # 费率最高的做空，费率最低的做多
            if middle["rate"] > 0:
                # 中间币也是正费率，做空
                print(f"  {lowest['symbol']}: LONG  $4000 (费率最低 {lowest['rate']*100:+.4f}%)")
                print(f"  {highest['symbol']}: SHORT $2000 (费率最高 {highest['rate']*100:+.4f}%)")
                print(f"  {middle['symbol']}: SHORT $2000 (费率 {middle['rate']*100:+.4f}%)")
            else:
                # 中间币负费率，做多
                print(f"  {lowest['symbol']}: LONG  $2000 (费率最低 {lowest['rate']*100:+.4f}%)")
                print(f"  {middle['symbol']}: LONG  $2000 (费率 {middle['rate']*100:+.4f}%)")
                print(f"  {highest['symbol']}: SHORT $4000 (费率最高 {highest['rate']*100:+.4f}%)")
        else:
            print(f"  {lowest['symbol']}: LONG")
            print(f"  {highest['symbol']}: SHORT")


if __name__ == "__main__":
    main()
