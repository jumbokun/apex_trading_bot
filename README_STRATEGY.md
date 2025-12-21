# Apex Trading Bot - 自动化交易策略系统

一个稳健的、基于量价分析的自动化交易系统，专为Apex Exchange设计。

## 核心特性

### 🛡️ 多层安全防护

1. **爆仓保护系统**
   - 实时监控保证金率
   - 当保证金率低于20%时紧急平仓
   - 远高于交易所爆仓线，确保安全

2. **多级止损机制**
   - 固定止损：默认2%
   - 追踪止损：锁定利润，默认3%
   - 智能止盈：默认5%

3. **仓位风控**
   - 单个仓位最大占比限制（默认10%）
   - 最大杠杆限制（默认3倍）
   - 动态仓位计算，根据信号强度调整

4. **每日风险限制**
   - 每日最大亏损限制（默认5%）
   - 达到限制自动停止交易
   - 每日最大交易次数限制

5. **市场环境过滤**
   - 避免在极端波动时交易
   - 成交量过滤，确保流动性
   - 资金费率监控，避免高费用

### 📊 智能交易策略

1. **量价分析**
   - 基于成交量确认信号
   - 检测成交量激增
   - 过滤虚假突破

2. **趋势跟踪**
   - 双均线系统（10/30周期）
   - RSI超买超卖识别
   - 多因子综合评分

3. **仓位管理**
   - 智能加仓：盈利时小幅加仓
   - 智能减仓：亏损时及时减仓
   - 分批建仓，降低风险

### 🤖 自动化功能

1. **启动时自动扫描**
   - 同步现有持仓
   - 检查并设置止损止盈
   - 评估风险状况

2. **持续风险监控**
   - 24/7监控持仓
   - 自动执行止损止盈
   - 异常情况告警

3. **智能评价系统**
   - 实时评估账户状态
   - 分析持仓健康度
   - 提供优化建议

## 安装依赖

```bash
pip install numpy loguru httpx
```

## 快速开始

### 1. 配置API密钥

编辑 `strategy_example.py`，填入你的API密钥：

```python
config.api_key = "your_api_key_here"
config.secret_key = "your_secret_key_here"
config.passphrase = "your_passphrase_here"
```

### 2. 选择运行模式

#### 模式1: 安全保守策略（推荐新手）

```python
from strategy_config import create_safe_config
from apex_trading_bot import ApexExchange
from trading_engine import TradingEngine

config = create_safe_config()
config.api_key = "..."
config.secret_key = "..."
config.passphrase = "..."

with ApexExchange(...) as apex:
    engine = TradingEngine(apex, config)
    engine.start()  # 持续运行
```

特点：
- 低杠杆（2倍）
- 小仓位（最大5%）
- 严格止损（1.5%）
- 长时间框架（1小时）

#### 模式2: 自定义策略

```python
from strategy_config import TradingConfig, StrategyConfig, RiskConfig

config = TradingConfig(
    strategy=StrategyConfig(
        symbols=["BTC-USDC", "ETH-USDC"],
        timeframe="15m",
        min_signal_strength=0.65,
    ),
    risk=RiskConfig(
        max_position_size=0.10,
        stop_loss_pct=0.02,
        max_daily_loss_pct=0.05,
    ),
)
```

#### 模式3: 仅风险检查（适合睡前运行）

```python
from risk_manager import RiskManager
from position_manager import PositionManager

# 只监控现有仓位，不开新仓
position_manager = PositionManager(apex, config.strategy, config.risk)
risk_manager = RiskManager(apex, position_manager, config.risk, dry_run=False)

# 执行风险检查和必要的平仓
actions = risk_manager.monitor_positions()
```

### 3. 运行

```bash
python strategy_example.py
```

## 配置说明

### 风险配置 (RiskConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_position_size` | 0.10 | 单个仓位最大占账户资金比例（10%） |
| `max_leverage` | 3.0 | 最大杠杆倍数 |
| `stop_loss_pct` | 0.02 | 固定止损百分比（2%） |
| `take_profit_pct` | 0.05 | 止盈百分比（5%） |
| `trailing_stop_pct` | 0.03 | 追踪止损百分比（3%） |
| `min_margin_ratio` | 0.15 | 最小保证金率（15%），低于此值警告 |
| `emergency_close_margin_ratio` | 0.20 | 紧急平仓保证金率（20%） |
| `max_daily_loss_pct` | 0.05 | 每日最大亏损（5%） |
| `max_daily_trades` | 20 | 每日最大交易次数 |
| `max_volatility` | 0.10 | 最大允许波动率（10%） |
| `min_volume_ratio` | 0.5 | 最小成交量比率（50%） |
| `max_funding_rate` | 0.0005 | 最大资金费率（0.05%） |

### 策略配置 (StrategyConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `symbols` | `["BTC-USDC"]` | 交易对列表 |
| `timeframe` | "15m" | K线周期 |
| `kline_lookback` | 100 | 回看K线数量 |
| `ma_short_period` | 10 | 短期均线周期 |
| `ma_long_period` | 30 | 长期均线周期 |
| `rsi_period` | 14 | RSI周期 |
| `rsi_overbought` | 70 | RSI超买阈值 |
| `rsi_oversold` | 30 | RSI超卖阈值 |
| `volume_ma_period` | 20 | 成交量均线周期 |
| `volume_surge_ratio` | 1.5 | 成交量激增比率 |
| `min_signal_strength` | 0.6 | 最小信号强度（0-1） |
| `initial_position_pct` | 0.03 | 初始仓位占比（3%） |

## 使用场景

### 场景1: 日常自动交易

```python
# 设置为实盘模式，持续运行
config.dry_run = False
config.scan_interval = 60  # 每60秒扫描一次

engine = TradingEngine(apex, config)
engine.start()  # 会持续运行直到手动停止
```

### 场景2: 睡觉前检查风险

```python
# 只检查现有持仓，确保安全
config.dry_run = False  # 实盘，可以真实平仓

position_manager.sync_positions()
risk_manager.monitor_positions()  # 会自动执行止损止盈
```

### 场景3: 回测或测试

```python
# 模拟模式，不会真实下单
config.dry_run = True
config.testnet = True

engine = TradingEngine(apex, config)
engine.run_once()  # 只运行一次
```

## 安全保障

### ✅ 可以放心2-3天的原因

1. **自动止损**
   - 每个持仓都有固定止损价格
   - 追踪止损锁定利润
   - 亏损超过设定值自动平仓

2. **爆仓保护**
   - 保证金率低于20%强制平仓
   - 远高于交易所爆仓线（通常5-10%）
   - 多重检查机制

3. **每日亏损限制**
   - 达到每日最大亏损自动停止交易
   - 防止连续亏损

4. **市场环境过滤**
   - 极端波动时不交易
   - 成交量不足时不交易
   - 避免被市场异常影响

5. **状态持久化**
   - 程序重启后恢复状态
   - 持仓信息不会丢失

### ⚠️ 注意事项

1. **首次使用建议**
   - 先在测试网测试
   - 使用模拟模式（dry_run=True）
   - 从小仓位开始

2. **定期检查**
   - 虽然可以放2-3天，但建议每天至少检查一次
   - 关注市场重大事件

3. **网络稳定性**
   - 确保服务器网络稳定
   - 建议使用VPS运行

4. **API密钥安全**
   - 使用只读+交易权限，不要给提现权限
   - 设置API IP白名单

## 监控和日志

### 日志文件

程序会自动生成日志文件：
- 位置：`logs/trading_YYYY-MM-DD.log`
- 每天轮换，保留30天
- 自动压缩旧日志

### 实时监控

程序会定期输出报告，包括：
- 账户余额
- 持仓情况
- 未实现盈亏
- 今日交易统计
- 风险评估

示例输出：
```
============================================================
交易报告
============================================================
账户余额: 10000.00 USDC
持仓数量: 1
总持仓价值: 1000.00 USDC
总未实现盈亏: 50.00 USDC
仓位占比: 10.00%
今日交易次数: 3
今日盈亏: 80.00 USDC

持仓详情:
  BTC-USDC LONG 0.010000 - 开仓价: 95000.00, 当前价: 100000.00, 盈亏: 50.00 (5.26%)
    保证金率: 35.50%
    止损: 93100.00, 止盈: 99750.00, 追踪止损: 97000.00

自动评价:
  ✓ 仓位占比保守
  ✓ 今日盈利 80.00 USDC
  ✓ 今日交易次数保守: 3/20
  ✓ BTC-USDC 表现良好，盈利 5.26%
============================================================
```

## 常见问题

### Q1: 程序意外停止怎么办？

A: 程序会保存状态到文件，重启后会自动恢复。现有持仓不会受影响。

### Q2: 如何紧急平掉所有仓位？

A: 可以直接在Apex交易所网页端操作，或运行：
```python
risk_manager._close_all_positions("手动紧急平仓")
```

### Q3: 如何调整止损价格？

A: 止损价格在开仓时自动设置，如需修改可以编辑 `RiskConfig` 中的 `stop_loss_pct`。

### Q4: 可以同时运行多个策略吗？

A: 建议不要同时运行多个实例交易同一个账户，可能会冲突。如需多策略，建议使用多个子账户。

### Q5: 如何知道策略表现如何？

A: 查看每日盈亏统计和自动评价系统的输出。

## 免责声明

本交易系统仅供学习和研究使用。加密货币交易具有高风险，可能导致资金损失。使用本系统进行实盘交易的所有风险由用户自行承担。

**重要提醒：**
- 不要投入超过你能承受损失的资金
- 充分理解策略逻辑再使用
- 建议先在测试网充分测试
- 定期检查持仓和账户状态

## 技术支持

如有问题，请查看日志文件或联系开发者。

---

祝交易顺利！🚀
