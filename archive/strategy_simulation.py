"""
策略模拟器 - 模拟各种市场情况下的策略运行
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SimulationState:
    """模拟状态"""
    prices: Dict[str, float]
    funding_rates: Dict[str, float]
    positions: Dict[str, dict]  # {symbol: {side, notional}}
    target_per_side: float = 50000.0


def print_state(state: SimulationState, title: str):
    """打印状态"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    print("\n[市场数据]")
    print(f"{'币种':<12} {'价格':>12} {'资金费率':>12} {'方向建议':>10}")
    print("-" * 50)

    sorted_rates = sorted(state.funding_rates.items(), key=lambda x: x[1], reverse=True)
    for symbol, rate in sorted_rates:
        price = state.prices.get(symbol, 0)
        direction = "SHORT" if rate > 0 else "LONG" if rate < 0 else "中性"
        print(f"{symbol:<12} ${price:>10,.2f} {rate*100:>+10.4f}% {direction:>10}")

    print("\n[当前持仓]")
    total_long = 0
    total_short = 0
    for symbol, pos in state.positions.items():
        if pos["side"]:
            side_str = pos["side"].value
            notional = pos["notional"]
            if pos["side"] == PositionSide.LONG:
                total_long += notional
            else:
                total_short += notional
            print(f"  {symbol}: {side_str} ${notional:,.0f}")
        else:
            print(f"  {symbol}: 无仓位")

    net_delta = total_long - total_short
    total = total_long + total_short
    imbalance = abs(net_delta) / total * 100 if total > 0 else 0

    print(f"\n  多头总计: ${total_long:,.0f}")
    print(f"  空头总计: ${total_short:,.0f}")
    print(f"  净Delta: ${net_delta:,.0f} (偏差 {imbalance:.1f}%)")
    print(f"  状态: {'[OK] 中性' if imbalance < 2 else '[!] 失衡'}")


def calculate_target(state: SimulationState) -> Dict[str, dict]:
    """根据费率计算目标配置"""
    sorted_rates = sorted(state.funding_rates.items(), key=lambda x: x[1], reverse=True)
    target = state.target_per_side

    if len(sorted_rates) >= 3:
        highest = sorted_rates[0]  # 做空
        middle = sorted_rates[1]
        lowest = sorted_rates[2]   # 做多

        if middle[1] > 0:
            # 中间也是正费率，两个空头
            return {
                lowest[0]: {"side": PositionSide.LONG, "notional": target},
                highest[0]: {"side": PositionSide.SHORT, "notional": target / 2},
                middle[0]: {"side": PositionSide.SHORT, "notional": target / 2}
            }
        else:
            # 中间是负费率
            return {
                lowest[0]: {"side": PositionSide.LONG, "notional": target / 2},
                middle[0]: {"side": PositionSide.LONG, "notional": target / 2},
                highest[0]: {"side": PositionSide.SHORT, "notional": target}
            }

    return {}


def print_target(state: SimulationState):
    """打印目标配置"""
    target = calculate_target(state)
    print("\n[目标配置] (根据费率自动计算)")
    for symbol, t in target.items():
        print(f"  {symbol}: {t['side'].value} ${t['notional']:,.0f}")


def calculate_actions(state: SimulationState) -> List[dict]:
    """计算需要执行的操作"""
    target = calculate_target(state)
    actions = []

    for symbol, t in target.items():
        current = state.positions.get(symbol, {"side": None, "notional": 0})

        # 需要反向
        if current["side"] and current["side"] != t["side"] and current["notional"] > 100:
            actions.append({
                "symbol": symbol,
                "action": "平仓",
                "from_side": current["side"].value,
                "notional": current["notional"]
            })
            actions.append({
                "symbol": symbol,
                "action": "开仓",
                "to_side": t["side"].value,
                "notional": t["notional"]
            })
        # 加仓或减仓
        elif current["side"] == t["side"] or current["notional"] < 100:
            diff = t["notional"] - current["notional"]
            if abs(diff) >= 100:
                if diff > 0:
                    actions.append({
                        "symbol": symbol,
                        "action": "加仓",
                        "side": t["side"].value,
                        "notional": diff
                    })
                else:
                    actions.append({
                        "symbol": symbol,
                        "action": "减仓",
                        "side": t["side"].value,
                        "notional": abs(diff)
                    })

    return actions


def print_actions(actions: List[dict]):
    """打印操作"""
    if not actions:
        print("\n[操作] 无需调仓")
        return

    print("\n[需要执行的操作]")
    for i, action in enumerate(actions, 1):
        if action["action"] == "平仓":
            print(f"  {i}. {action['symbol']}: 平仓 {action['from_side']} ${action['notional']:,.0f}")
        elif action["action"] == "开仓":
            print(f"  {i}. {action['symbol']}: 开仓 {action['to_side']} ${action['notional']:,.0f}")
        elif action["action"] == "加仓":
            print(f"  {i}. {action['symbol']}: 加仓 {action['side']} +${action['notional']:,.0f}")
        elif action["action"] == "减仓":
            print(f"  {i}. {action['symbol']}: 减仓 {action['side']} -${action['notional']:,.0f}")


def simulate_batched_execution(actions: List[dict], batch_count: int = 10):
    """模拟分批执行"""
    if not actions:
        return

    total_notional = sum(a.get("notional", 0) for a in actions)
    batch_size = total_notional / batch_count

    print(f"\n[分批执行计划]")
    print(f"  总调仓金额: ${total_notional:,.0f}")
    print(f"  分批次数: {batch_count}")
    print(f"  每批金额: ~${batch_size:,.0f}")
    print(f"  每批间隔: 60秒")
    print(f"  预计完成时间: ~{batch_count}分钟")


def run_simulation():
    """运行各种情况的模拟"""

    print("\n" + "=" * 70)
    print("   自适应Delta中性策略 - 情景模拟")
    print("=" * 70)
    print("\n目标配置: 多头 $50,000 + 空头 $50,000 = 总敞口 $100,000")
    print("币种: BTC-USDT, ETH-USDT, SOL-USDT")
    print("策略: 费率最高做空，费率最低做多")

    # ========== 情景1: 初始建仓 ==========
    print("\n\n" + "#" * 70)
    print("#  情景1: 初始建仓 (无持仓)")
    print("#" * 70)

    state1 = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 190},
        funding_rates={"BTC-USDT": 0.0008, "ETH-USDT": 0.0012, "SOL-USDT": 0.0003},
        positions={"BTC-USDT": {"side": None, "notional": 0},
                   "ETH-USDT": {"side": None, "notional": 0},
                   "SOL-USDT": {"side": None, "notional": 0}}
    )

    print_state(state1, "初始状态")
    print_target(state1)
    actions1 = calculate_actions(state1)
    print_actions(actions1)
    simulate_batched_execution(actions1)

    print("\n[结论]")
    print("  ETH费率最高(+0.0012%) -> 做空")
    print("  BTC费率次高(+0.0008%) -> 做空")
    print("  SOL费率最低(+0.0003%) -> 做多")
    print("  配置: SOL多$50K / ETH空$25K + BTC空$25K")

    # ========== 情景2: 费率变化导致方向调整 ==========
    print("\n\n" + "#" * 70)
    print("#  情景2: 费率变化 - ETH从正变负")
    print("#" * 70)

    state2_before = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 190},
        funding_rates={"BTC-USDT": 0.0008, "ETH-USDT": 0.0012, "SOL-USDT": 0.0003},
        positions={"BTC-USDT": {"side": PositionSide.SHORT, "notional": 25000},
                   "ETH-USDT": {"side": PositionSide.SHORT, "notional": 25000},
                   "SOL-USDT": {"side": PositionSide.LONG, "notional": 50000}}
    )
    print_state(state2_before, "变化前 (原配置)")

    state2_after = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 190},
        funding_rates={"BTC-USDT": 0.0010, "ETH-USDT": -0.0005, "SOL-USDT": 0.0008},  # ETH变负!
        positions=state2_before.positions.copy()
    )
    print_state(state2_after, "变化后 (费率变化)")
    print_target(state2_after)
    actions2 = calculate_actions(state2_after)
    print_actions(actions2)
    simulate_batched_execution(actions2)

    print("\n[结论]")
    print("  ETH费率变负(-0.0005%) -> 从空头改为多头!")
    print("  BTC费率最高(+0.0010%) -> 空头")
    print("  新配置: ETH多$25K + SOL多$25K / BTC空$50K")
    print("  需要: ETH平空$25K再开多$25K (共调仓$50K)")
    print("  执行: 分10批，每批$5K，每分钟1批，约10分钟完成")

    # ========== 情景3: 价格波动导致Delta失衡 ==========
    print("\n\n" + "#" * 70)
    print("#  情景3: BTC价格大涨 - Delta失衡")
    print("#" * 70)

    state3_before = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 190},
        funding_rates={"BTC-USDT": 0.0008, "ETH-USDT": 0.0012, "SOL-USDT": 0.0003},
        positions={"BTC-USDT": {"side": PositionSide.SHORT, "notional": 25000},
                   "ETH-USDT": {"side": PositionSide.SHORT, "notional": 25000},
                   "SOL-USDT": {"side": PositionSide.LONG, "notional": 50000}}
    )
    print_state(state3_before, "价格波动前")

    # BTC涨10%，空头亏损，仓位变化
    state3_after = SimulationState(
        prices={"BTC-USDT": 104500, "ETH-USDT": 3300, "SOL-USDT": 190},  # BTC +10%
        funding_rates={"BTC-USDT": 0.0008, "ETH-USDT": 0.0012, "SOL-USDT": 0.0003},
        positions={
            "BTC-USDT": {"side": PositionSide.SHORT, "notional": 27500},  # 空头仓位增加(亏损)
            "ETH-USDT": {"side": PositionSide.SHORT, "notional": 25000},
            "SOL-USDT": {"side": PositionSide.LONG, "notional": 50000}
        }
    )
    print_state(state3_after, "BTC涨10%后 (Delta失衡)")

    # 计算Delta
    total_long = 50000
    total_short = 27500 + 25000  # 52500
    net_delta = total_long - total_short  # -2500
    print(f"\n[Delta分析]")
    print(f"  多头: ${total_long:,} | 空头: ${total_short:,}")
    print(f"  净Delta: ${net_delta:,} (偏空 {abs(net_delta)/total_long*100:.1f}%)")
    print(f"  需要: 减少空头或增加多头 ${abs(net_delta):,}")

    print("\n[自动调仓操作]")
    print(f"  1. BTC-USDT: 减空 ${abs(net_delta)/2:,.0f} (买入平仓)")
    print(f"  2. SOL-USDT: 加多 ${abs(net_delta)/2:,.0f} (买入开仓)")
    print(f"  调仓后恢复Delta中性")

    # ========== 情景4: 紧急情况 - 极端价格波动 ==========
    print("\n\n" + "#" * 70)
    print("#  情景4: 极端波动 - SOL暴跌20%")
    print("#" * 70)

    state4 = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 152},  # SOL -20%
        funding_rates={"BTC-USDT": 0.0008, "ETH-USDT": 0.0012, "SOL-USDT": -0.0015},  # SOL费率变极负
        positions={
            "BTC-USDT": {"side": PositionSide.SHORT, "notional": 25000},
            "ETH-USDT": {"side": PositionSide.SHORT, "notional": 25000},
            "SOL-USDT": {"side": PositionSide.LONG, "notional": 40000}  # 多头亏损，仓位缩水
        }
    )
    print_state(state4, "SOL暴跌20%后")

    total_long = 40000
    total_short = 50000
    net_delta = total_long - total_short
    imbalance = abs(net_delta) / (total_long + total_short) * 100

    print(f"\n[紧急情况分析]")
    print(f"  Delta偏差: {imbalance:.1f}% > 5% 紧急阈值!")
    print(f"  触发: 紧急调仓模式")
    print(f"\n[紧急调仓]")
    print(f"  方案A: 加多SOL ${abs(net_delta):,} (补仓)")
    print(f"  方案B: 减空ETH或BTC ${abs(net_delta):,} (平仓)")
    print(f"  执行: 立即用市价单完成，不分批")

    # ========== 情景5: 所有费率都是负的 ==========
    print("\n\n" + "#" * 70)
    print("#  情景5: 熊市 - 所有费率为负")
    print("#" * 70)

    state5 = SimulationState(
        prices={"BTC-USDT": 85000, "ETH-USDT": 2800, "SOL-USDT": 150},
        funding_rates={"BTC-USDT": -0.0003, "ETH-USDT": -0.0008, "SOL-USDT": -0.0012},  # 全负
        positions={"BTC-USDT": {"side": None, "notional": 0},
                   "ETH-USDT": {"side": None, "notional": 0},
                   "SOL-USDT": {"side": None, "notional": 0}}
    )
    print_state(state5, "熊市状态 (全负费率)")
    print_target(state5)

    print("\n[策略分析]")
    print("  所有费率为负 -> 多头有利")
    print("  BTC费率最高(-0.0003%) -> 做空 (少付)")
    print("  SOL费率最低(-0.0012%) -> 做多 (收取最多)")
    print("  配置: SOL多$25K + ETH多$25K / BTC空$50K")
    print("\n  注意: 即使全负，也要选相对最高的做空，保持对冲!")

    # ========== 情景6: 结算前检查 ==========
    print("\n\n" + "#" * 70)
    print("#  情景6: 结算前5分钟 - 费率变化")
    print("#" * 70)

    print("\n[时间] 距离下次结算: 4分32秒")
    print("[触发] 结算前自动检查费率")

    state6_before = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 190},
        funding_rates={"BTC-USDT": 0.0008, "ETH-USDT": 0.0012, "SOL-USDT": 0.0003},
        positions={"BTC-USDT": {"side": PositionSide.SHORT, "notional": 25000},
                   "ETH-USDT": {"side": PositionSide.SHORT, "notional": 25000},
                   "SOL-USDT": {"side": PositionSide.LONG, "notional": 50000}}
    )

    state6_after = SimulationState(
        prices={"BTC-USDT": 95000, "ETH-USDT": 3300, "SOL-USDT": 190},
        funding_rates={"BTC-USDT": 0.0015, "ETH-USDT": 0.0005, "SOL-USDT": 0.0002},  # BTC费率飙升
        positions=state6_before.positions.copy()
    )

    print("\n[费率变化]")
    print(f"  BTC: +0.0008% -> +0.0015% (上涨)")
    print(f"  ETH: +0.0012% -> +0.0005% (下降)")
    print(f"  SOL: +0.0003% -> +0.0002% (下降)")

    print("\n[决策]")
    print("  费率排序变化: ETH最高 -> BTC最高")
    print("  但距离结算仅4分钟，来不及调仓")
    print("  策略: 等待本次结算后再调整")
    print("  原因: 调仓可能错过本次结算收益")

    # ========== 总结 ==========
    print("\n\n" + "=" * 70)
    print("   策略行为总结")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        自适应策略行为表                              │
├──────────────────┬──────────────────────────────────────────────────┤
│ 触发条件          │ 策略行为                                         │
├──────────────────┼──────────────────────────────────────────────────┤
│ 初始建仓          │ 根据费率排序，费率最低做多，最高做空              │
│                  │ 分10批建仓，每批间隔60秒                          │
├──────────────────┼──────────────────────────────────────────────────┤
│ 费率变化 > 0.05%  │ 重新计算最优配置                                  │
│                  │ 如需反向，分批平仓再开仓                          │
│                  │ 每批5%仓位，限价单执行                            │
├──────────────────┼──────────────────────────────────────────────────┤
│ Delta偏差 > 2%    │ 常规调仓：加多或减空恢复平衡                      │
│                  │ 分批执行，每批间隔60秒                            │
├──────────────────┼──────────────────────────────────────────────────┤
│ Delta偏差 > 5%    │ 紧急调仓：立即执行                                │
│                  │ 使用市价单确保成交                                │
├──────────────────┼──────────────────────────────────────────────────┤
│ 结算前5分钟       │ 检查费率，评估是否需要调整                        │
│                  │ 如需大幅调整，等待结算后再执行                    │
├──────────────────┼──────────────────────────────────────────────────┤
│ 限价单超时30秒    │ 取消重挂，价格更接近市价                          │
│                  │ 重试3次后改用市价单                               │
└──────────────────┴──────────────────────────────────────────────────┘

费率方向选择逻辑:
┌─────────────────────────────────────────────────────────────────────┐
│ 费率排序 (高→低)  │  BTC > ETH > SOL                                 │
│ 方向分配          │  BTC空 / ETH空 / SOL多  (如中间>0)               │
│                  │  BTC空 / ETH多 / SOL多  (如中间<0)               │
├─────────────────────────────────────────────────────────────────────┤
│ 示例1: +0.12% > +0.08% > +0.03%                                     │
│ 结果:  ETH空$25K + BTC空$25K / SOL多$50K                            │
├─────────────────────────────────────────────────────────────────────┤
│ 示例2: +0.10% > -0.02% > -0.05%                                     │
│ 结果:  BTC空$50K / ETH多$25K + SOL多$25K                            │
└─────────────────────────────────────────────────────────────────────┘

波动率自适应规则 (Delta中性 = 低风险 = 高杠杆):
┌─────────────────────────────────────────────────────────────────────┐
│ 用户设定: 最大仓位 $50,000/边 | 最小仓位 $10,000/边                  │
│          杠杆范围 20x-50x (Delta中性风险低)                          │
├──────────────────┬──────────────────────────────────────────────────┤
│ 24h波动率         │ 自适应调整                                       │
├──────────────────┼──────────────────────────────────────────────────┤
│ < 2% (低波动)     │ 杠杆: 50x | 仓位: $50,000 (100%)                 │
│ 2%-5% (中波动)    │ 杠杆: 30x-50x | 仓位: $30,000-$50,000 (60%-100%) │
│ 5%-8% (高波动)    │ 杠杆: 20x-30x | 仓位: $10,000-$30,000 (20%-60%)  │
│ > 8% (极端)       │ 杠杆: 20x | 仓位: $10,000 (20% 最小)             │
└──────────────────┴──────────────────────────────────────────────────┘

示例:
  BTC 24h波动率: 3.5% (中等)
  ETH 24h波动率: 4.2% (中等)
  SOL 24h波动率: 6.8% (较高)
  平均波动率: 4.8%
  -> 自适应杠杆: 32x
  -> 自适应仓位: $32,000/边 (64%)
""")


if __name__ == "__main__":
    run_simulation()
