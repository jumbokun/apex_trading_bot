"""
回测分析：Delta中性策略资金费率收益
计算历史资金费率收益、手续费成本和净收益
"""
import urllib.request
import json
import ssl
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import time


@dataclass
class FundingRecord:
    """资金费率记录"""
    symbol: str
    rate: float
    price: float
    timestamp: datetime


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: str  # LONG or SHORT
    notional: float  # 名义价值 USD


def fetch_historical_funding_rates(symbol: str, limit: int = 100, page: int = 0) -> List[FundingRecord]:
    """获取历史资金费率"""
    endpoint = "https://omni.apex.exchange"
    url = f"{endpoint}/api/v3/history-funding?symbol={symbol}&limit={limit}&page={page}"

    ssl_ctx = ssl.create_default_context()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
            # API 返回格式: {"data": {"historyFunds": [...]}}
            history_data = data.get("data", data)  # 兼容两种格式
            records = []
            for item in history_data.get("historyFunds", []):
                try:
                    # 时间戳可能是毫秒或秒
                    ts = item.get("fundingTimestamp", 0)
                    if ts > 10000000000:  # 毫秒
                        ts = ts / 1000
                    dt = datetime.fromtimestamp(ts)

                    records.append(FundingRecord(
                        symbol=item.get("symbol", symbol),
                        rate=float(item.get("rate", 0)),
                        price=float(item.get("price", 0)),
                        timestamp=dt
                    ))
                except Exception as e:
                    print(f"解析记录失败: {e}")
            return records
    except Exception as e:
        print(f"获取 {symbol} 历史费率失败: {e}")
        return []


def fetch_all_funding_rates(symbol: str, days: int = 30) -> List[FundingRecord]:
    """获取指定天数的所有历史资金费率"""
    all_records = []
    page = 0
    limit = 100
    target_time = datetime.now() - timedelta(days=days)

    while True:
        records = fetch_historical_funding_rates(symbol, limit=limit, page=page)
        if not records:
            break

        all_records.extend(records)

        # 检查是否已经获取到目标时间之前的数据
        oldest = min(r.timestamp for r in records)
        if oldest < target_time:
            break

        page += 1
        time.sleep(0.2)  # 避免请求过快

    # 过滤到目标时间范围内
    filtered = [r for r in all_records if r.timestamp >= target_time]
    return sorted(filtered, key=lambda x: x.timestamp)


def calculate_funding_income(positions: List[Position],
                             funding_records: Dict[str, List[FundingRecord]]) -> Dict:
    """
    计算资金费率收入

    正费率:
    - 多头支付给空头
    - 空头收取费率

    负费率:
    - 空头支付给多头
    - 多头收取费率
    """
    results = {
        "total_income": 0.0,
        "by_symbol": {},
        "settlements": 0,
        "details": []
    }

    for pos in positions:
        symbol = pos.symbol
        records = funding_records.get(symbol, [])

        symbol_income = 0.0
        for record in records:
            # 计算本次结算的收益
            # 正费率: 空头收取
            # 负费率: 多头收取
            if pos.side == "SHORT":
                # 空头: 正费率收取，负费率支付
                income = pos.notional * record.rate
            else:
                # 多头: 正费率支付，负费率收取
                income = -pos.notional * record.rate

            symbol_income += income
            results["details"].append({
                "time": record.timestamp,
                "symbol": symbol,
                "side": pos.side,
                "rate": record.rate,
                "notional": pos.notional,
                "income": income
            })

        results["by_symbol"][symbol] = {
            "side": pos.side,
            "notional": pos.notional,
            "total_income": symbol_income,
            "settlements": len(records),
            "avg_rate": sum(r.rate for r in records) / len(records) if records else 0
        }
        results["total_income"] += symbol_income
        results["settlements"] = max(results["settlements"], len(records))

    return results


def calculate_rebalance_costs(positions: List[Position],
                              days: int,
                              delta_threshold: float = 0.05,
                              maker_fee: float = 0.0002,
                              taker_fee: float = 0.0005,
                              use_maker: bool = True) -> Dict:
    """
    估算调仓成本

    假设:
    - 每天价格波动导致一定次数的调仓
    - 平均每次调仓涉及总仓位的 5-10%
    """
    total_notional = sum(p.notional for p in positions)

    # 估算调仓频率 (保守估计: 每天 2-4 次)
    rebalances_per_day = 3
    total_rebalances = rebalances_per_day * days

    # 每次调仓涉及的金额 (假设平均调整 5% 的仓位)
    avg_rebalance_amount = total_notional * 0.05

    # 总交易量
    total_volume = avg_rebalance_amount * total_rebalances * 2  # 买卖两边

    # 手续费
    fee_rate = maker_fee if use_maker else taker_fee
    total_fees = total_volume * fee_rate

    return {
        "total_volume": total_volume,
        "total_fees": total_fees,
        "rebalances": total_rebalances,
        "avg_rebalance_amount": avg_rebalance_amount,
        "fee_rate": fee_rate
    }


def calculate_initial_position_cost(positions: List[Position],
                                     maker_fee: float = 0.0002,
                                     taker_fee: float = 0.0005,
                                     use_maker: bool = True) -> Dict:
    """计算初始建仓成本"""
    total_notional = sum(p.notional for p in positions)
    fee_rate = maker_fee if use_maker else taker_fee

    return {
        "total_notional": total_notional,
        "fees": total_notional * fee_rate,
        "fee_rate": fee_rate
    }


def run_backtest(days: int = 7,
                 btc_long: float = 10000.0,
                 eth_short: float = 5000.0,
                 sol_short: float = 5000.0,
                 use_maker: bool = True):
    """
    运行回测

    Args:
        days: 回测天数
        btc_long: BTC 多头仓位 (USD)
        eth_short: ETH 空头仓位 (USD)
        sol_short: SOL 空头仓位 (USD)
        use_maker: 是否使用 Maker 费率
    """
    print("="*70)
    print(f"Delta中性策略回测分析 - 过去 {days} 天")
    print("="*70)

    # 定义持仓
    positions = [
        Position("BTC-USDT", "LONG", btc_long),
        Position("ETH-USDT", "SHORT", eth_short),
        Position("SOL-USDT", "SHORT", sol_short),
    ]

    total_notional = sum(p.notional for p in positions)
    print(f"\n持仓配置:")
    for p in positions:
        print(f"  {p.symbol}: {p.side} ${p.notional:,.0f}")
    print(f"  总仓位: ${total_notional:,.0f}")

    # 获取历史资金费率
    print(f"\n正在获取历史资金费率 (过去{days}天)...")
    funding_records = {}
    for p in positions:
        print(f"  获取 {p.symbol}...")
        records = fetch_all_funding_rates(p.symbol, days=days)
        funding_records[p.symbol] = records
        if records:
            print(f"    获取到 {len(records)} 条记录")
            print(f"    时间范围: {records[0].timestamp} ~ {records[-1].timestamp}")
        else:
            print(f"    警告: 无数据")

    # 计算资金费率收入
    print("\n计算资金费率收益...")
    funding_results = calculate_funding_income(positions, funding_records)

    # 计算调仓成本
    rebalance_costs = calculate_rebalance_costs(positions, days, use_maker=use_maker)

    # 计算初始建仓成本
    initial_costs = calculate_initial_position_cost(positions, use_maker=use_maker)

    # 输出结果
    print("\n" + "="*70)
    print("资金费率收益明细")
    print("="*70)

    for symbol, data in funding_results["by_symbol"].items():
        direction = "收取" if data["total_income"] > 0 else "支付"
        print(f"\n{symbol} ({data['side']}):")
        print(f"  仓位: ${data['notional']:,.0f}")
        print(f"  结算次数: {data['settlements']}")
        print(f"  平均费率: {data['avg_rate']*100:.6f}%")
        print(f"  费率收益: ${data['total_income']:+,.2f} ({direction})")

    print(f"\n资金费率总收益: ${funding_results['total_income']:+,.2f}")

    print("\n" + "="*70)
    print("交易成本估算")
    print("="*70)

    print(f"\n初始建仓:")
    print(f"  总仓位: ${initial_costs['total_notional']:,.0f}")
    print(f"  费率: {initial_costs['fee_rate']*100:.3f}% ({'Maker' if use_maker else 'Taker'})")
    print(f"  手续费: ${initial_costs['fees']:,.2f}")

    print(f"\n调仓成本 (估算):")
    print(f"  调仓次数: {rebalance_costs['rebalances']} 次")
    print(f"  平均调仓金额: ${rebalance_costs['avg_rebalance_amount']:,.0f}")
    print(f"  总交易量: ${rebalance_costs['total_volume']:,.0f}")
    print(f"  手续费: ${rebalance_costs['total_fees']:,.2f}")

    total_fees = initial_costs['fees'] + rebalance_costs['total_fees']
    print(f"\n总手续费: ${total_fees:,.2f}")

    # 净收益
    print("\n" + "="*70)
    print("净收益汇总")
    print("="*70)

    net_income = funding_results['total_income'] - total_fees

    print(f"\n资金费率收益: ${funding_results['total_income']:+,.2f}")
    print(f"手续费支出:   ${-total_fees:,.2f}")
    print(f"{'='*30}")
    print(f"净收益:       ${net_income:+,.2f}")

    # 年化收益率
    days_actual = days
    annual_return = (net_income / total_notional) * (365 / days_actual) * 100

    print(f"\n收益率分析:")
    print(f"  {days}天收益率: {(net_income/total_notional)*100:+.2f}%")
    print(f"  年化收益率: {annual_return:+.2f}%")

    # 每日明细 (最近 7 天)
    print("\n" + "="*70)
    print("每日资金费率明细 (最近数据)")
    print("="*70)

    # 按日期汇总
    daily_summary = {}
    for detail in funding_results["details"]:
        date_str = detail["time"].strftime("%Y-%m-%d")
        if date_str not in daily_summary:
            daily_summary[date_str] = {"income": 0, "settlements": 0}
        daily_summary[date_str]["income"] += detail["income"]
        daily_summary[date_str]["settlements"] += 1

    print(f"\n{'日期':<12} {'结算次数':>8} {'收益(USD)':>12}")
    print("-" * 35)
    sorted_dates = sorted(daily_summary.keys(), reverse=True)[:7]
    for date_str in reversed(sorted_dates):
        data = daily_summary[date_str]
        print(f"{date_str:<12} {data['settlements']//3:>8} ${data['income']:>+10.2f}")

    # 输出表格
    print("\n" + "="*70)
    print("回测结果汇总表")
    print("="*70)

    print(f"""
┌───────────────────────────────────────────────────────────────────┐
│                    Delta中性策略回测结果                           │
├───────────────────────────────────────────────────────────────────┤
│ 回测周期: {days} 天                                                │
│ 总仓位:   ${total_notional:>10,.0f}                                      │
├───────────────────────────────────────────────────────────────────┤
│ 持仓配置:                                                          │
│   BTC-USDT LONG  ${btc_long:>10,.0f}                                     │
│   ETH-USDT SHORT ${eth_short:>10,.0f}                                     │
│   SOL-USDT SHORT ${sol_short:>10,.0f}                                     │
├───────────────────────────────────────────────────────────────────┤
│ 收益:                                                              │
│   资金费率收益:  ${funding_results['total_income']:>+10.2f}                              │
│   总手续费:      ${-total_fees:>+10.2f}                              │
│   净收益:        ${net_income:>+10.2f}                              │
├───────────────────────────────────────────────────────────────────┤
│ 交易量:                                                            │
│   初始建仓:      ${total_notional:>10,.0f}                                      │
│   调仓交易量:    ${rebalance_costs['total_volume']:>10,.0f}                                      │
│   总交易量:      ${total_notional + rebalance_costs['total_volume']:>10,.0f}                                      │
├───────────────────────────────────────────────────────────────────┤
│ 收益率:                                                            │
│   {days}天收益率:     {(net_income/total_notional)*100:>+8.2f}%                                    │
│   年化收益率:    {annual_return:>+8.2f}%                                    │
└───────────────────────────────────────────────────────────────────┘
""")

    return {
        "days": days,
        "total_notional": total_notional,
        "funding_income": funding_results["total_income"],
        "total_fees": total_fees,
        "net_income": net_income,
        "total_volume": total_notional + rebalance_costs["total_volume"],
        "annual_return": annual_return
    }


if __name__ == "__main__":
    # 回测上周 (7天)
    print("\n" + "#"*70)
    print("# 上周回测 (7天)")
    print("#"*70)
    result_7d = run_backtest(days=7, btc_long=10000, eth_short=5000, sol_short=5000)

    # 回测一个月 (30天)
    print("\n\n" + "#"*70)
    print("# 一个月回测 (30天)")
    print("#"*70)
    result_30d = run_backtest(days=30, btc_long=10000, eth_short=5000, sol_short=5000)
