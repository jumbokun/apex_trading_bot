"""
回测过去30天每小时的波动率计算结果
使用与实盘相同的ATR14算法
"""
import json
import ssl
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict


# 配置参数 (与实盘一致)
VOL_LOW = 0.004   # 0.4% - 低波动阈值 (满仓)
VOL_HIGH = 0.012  # 1.2% - 高波动阈值 (最低仓)
MIN_NOTIONAL = 10000.0  # 最小仓位
MAX_NOTIONAL = 60000.0  # 最大仓位
ATR_PERIOD = 14  # ATR周期


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 500, end_time: int = None) -> list:
    """从Binance获取K线数据"""
    binance_symbol = symbol.replace("-", "")

    ssl_ctx = ssl.create_default_context()
    url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"
    if end_time:
        url += f"&endTime={end_time}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            klines = []
            for k in data:
                klines.append({
                    "time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return klines
    except Exception as e:
        print(f"获取K线失败 {symbol}: {e}")
        return []


def calculate_atr(klines: list, period: int = 14) -> float:
    """计算ATR (Average True Range)"""
    if len(klines) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(klines)):
        high = klines[i]['high']
        low = klines[i]['low']
        prev_close = klines[i-1]['close']

        # True Range = max(高-低, |高-前收|, |低-前收|)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # ATR = TR 的简单移动平均 (取最后period个)
    if len(true_ranges) < period:
        return 0.0

    atr = sum(true_ranges[-period:]) / period
    return atr


def calculate_atr_pct(klines: list, period: int = 14) -> float:
    """计算ATR百分比"""
    atr = calculate_atr(klines, period)
    if not klines or atr == 0:
        return 0.0

    current_price = klines[-1]['close']
    return atr / current_price if current_price > 0 else 0.0


def calculate_target_notional(vol_pct: float) -> float:
    """根据波动率计算目标仓位"""
    if vol_pct <= VOL_LOW:
        return MAX_NOTIONAL
    elif vol_pct >= VOL_HIGH:
        return MIN_NOTIONAL
    else:
        # 线性插值
        vol_score = (vol_pct - VOL_LOW) / (VOL_HIGH - VOL_LOW)
        return MAX_NOTIONAL - vol_score * (MAX_NOTIONAL - MIN_NOTIONAL)


def main():
    print("=" * 80)
    print("波动率回测分析 - 过去30天每小时")
    print("=" * 80)
    print(f"配置: VOL_LOW={VOL_LOW*100:.1f}%, VOL_HIGH={VOL_HIGH*100:.1f}%")
    print(f"仓位范围: ${MIN_NOTIONAL:,.0f} ~ ${MAX_NOTIONAL:,.0f}")
    print(f"ATR周期: {ATR_PERIOD}")
    print()

    # 获取30天的1小时K线数据
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    all_klines = {}

    print("获取历史K线数据...")
    for symbol in symbols:
        # 获取足够多的K线 (30天 * 24小时 + ATR周期)
        klines = fetch_binance_klines(symbol, "1h", limit=750)
        all_klines[symbol] = klines
        print(f"  {symbol}: {len(klines)} 条K线")

    if not all(all_klines.values()):
        print("获取数据失败")
        return

    # 确定时间范围
    btc_klines = all_klines["BTC-USDT"]
    start_idx = ATR_PERIOD + 1  # 需要足够的历史数据计算ATR

    # 存储结果
    results = []
    hourly_stats = defaultdict(lambda: {"count": 0, "total_vol": 0, "total_target": 0})
    daily_stats = defaultdict(lambda: {"count": 0, "total_vol": 0, "total_target": 0, "min_target": float('inf'), "max_target": 0})

    print(f"\n分析 {len(btc_klines) - start_idx} 个小时的数据...")
    print()

    # 遍历每个小时
    for i in range(start_idx, len(btc_klines)):
        # 截取到当前时间点的K线
        btc_slice = btc_klines[:i+1]

        # 计算BTC的ATR%
        btc_vol = calculate_atr_pct(btc_slice, ATR_PERIOD)
        target = calculate_target_notional(btc_vol)

        # 记录时间
        ts = btc_klines[i]['time']
        dt = datetime.fromtimestamp(ts / 1000)
        hour = dt.hour
        date_str = dt.strftime("%Y-%m-%d")

        results.append({
            "datetime": dt,
            "date": date_str,
            "hour": hour,
            "btc_vol": btc_vol,
            "target": target,
            "btc_price": btc_klines[i]['close']
        })

        # 按小时统计
        hourly_stats[hour]["count"] += 1
        hourly_stats[hour]["total_vol"] += btc_vol
        hourly_stats[hour]["total_target"] += target

        # 按日期统计
        daily_stats[date_str]["count"] += 1
        daily_stats[date_str]["total_vol"] += btc_vol
        daily_stats[date_str]["total_target"] += target
        daily_stats[date_str]["min_target"] = min(daily_stats[date_str]["min_target"], target)
        daily_stats[date_str]["max_target"] = max(daily_stats[date_str]["max_target"], target)

    # 输出每小时平均统计
    print("=" * 80)
    print("按小时统计 (UTC时间)")
    print("=" * 80)
    print(f"{'小时':<6} {'平均波动率':>12} {'平均目标仓位':>14} {'样本数':>8}")
    print("-" * 80)

    for hour in range(24):
        if hour in hourly_stats:
            stats = hourly_stats[hour]
            avg_vol = stats["total_vol"] / stats["count"]
            avg_target = stats["total_target"] / stats["count"]
            print(f"{hour:02d}:00  {avg_vol*100:>10.2f}%  ${avg_target:>12,.0f}  {stats['count']:>8}")

    # 输出每日统计
    print()
    print("=" * 80)
    print("按日期统计")
    print("=" * 80)
    print(f"{'日期':<12} {'平均波动率':>12} {'平均目标':>12} {'最小目标':>12} {'最大目标':>12}")
    print("-" * 80)

    for date in sorted(daily_stats.keys())[-30:]:  # 只显示最近30天
        stats = daily_stats[date]
        avg_vol = stats["total_vol"] / stats["count"]
        avg_target = stats["total_target"] / stats["count"]
        print(f"{date:<12} {avg_vol*100:>10.2f}%  ${avg_target:>10,.0f}  ${stats['min_target']:>10,.0f}  ${stats['max_target']:>10,.0f}")

    # 总体统计
    print()
    print("=" * 80)
    print("总体统计")
    print("=" * 80)

    all_vols = [r["btc_vol"] for r in results]
    all_targets = [r["target"] for r in results]

    avg_vol = sum(all_vols) / len(all_vols)
    min_vol = min(all_vols)
    max_vol = max(all_vols)

    avg_target = sum(all_targets) / len(all_targets)
    min_target = min(all_targets)
    max_target = max(all_targets)

    # 统计目标仓位分布
    target_dist = {
        f"${MIN_NOTIONAL:,.0f} (最低)": 0,
        f"${MIN_NOTIONAL:,.0f}-${(MIN_NOTIONAL+MAX_NOTIONAL)/2:,.0f}": 0,
        f"${(MIN_NOTIONAL+MAX_NOTIONAL)/2:,.0f}-${MAX_NOTIONAL:,.0f}": 0,
        f"${MAX_NOTIONAL:,.0f} (最高)": 0,
    }

    for t in all_targets:
        if t <= MIN_NOTIONAL:
            target_dist[f"${MIN_NOTIONAL:,.0f} (最低)"] += 1
        elif t >= MAX_NOTIONAL:
            target_dist[f"${MAX_NOTIONAL:,.0f} (最高)"] += 1
        elif t < (MIN_NOTIONAL + MAX_NOTIONAL) / 2:
            target_dist[f"${MIN_NOTIONAL:,.0f}-${(MIN_NOTIONAL+MAX_NOTIONAL)/2:,.0f}"] += 1
        else:
            target_dist[f"${(MIN_NOTIONAL+MAX_NOTIONAL)/2:,.0f}-${MAX_NOTIONAL:,.0f}"] += 1

    print(f"样本数: {len(results)} 小时")
    print(f"波动率: 平均 {avg_vol*100:.2f}%, 最小 {min_vol*100:.2f}%, 最大 {max_vol*100:.2f}%")
    print(f"目标仓位: 平均 ${avg_target:,.0f}, 最小 ${min_target:,.0f}, 最大 ${max_target:,.0f}")
    print()
    print("目标仓位分布:")
    for label, count in target_dist.items():
        pct = count / len(all_targets) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<25} {bar:<25} {count:>4} ({pct:.1f}%)")

    # 显示最近24小时详细数据
    print()
    print("=" * 80)
    print("最近24小时详细数据")
    print("=" * 80)
    print(f"{'时间':<20} {'BTC价格':>12} {'波动率(ATR14)':>14} {'目标仓位':>14}")
    print("-" * 80)

    for r in results[-24:]:
        vol_indicator = ""
        if r["btc_vol"] <= VOL_LOW:
            vol_indicator = " [低]"
        elif r["btc_vol"] >= VOL_HIGH:
            vol_indicator = " [高]"

        print(f"{r['datetime'].strftime('%Y-%m-%d %H:%M'):<20} ${r['btc_price']:>10,.0f} {r['btc_vol']*100:>12.2f}%{vol_indicator:<4} ${r['target']:>12,.0f}")


if __name__ == "__main__":
    main()
