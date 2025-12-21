# 快速入门指南

## 5分钟快速开始

### 1. 安装依赖

```bash
cd apex_trading_bot
pip install -r requirements_strategy.txt
```

### 2. 配置API密钥

有两种方式：

#### 方式A: 环境变量（推荐）

```bash
# Linux/Mac
export APEX_API_KEY="your_api_key"
export APEX_SECRET_KEY="your_secret_key"
export APEX_PASSPHRASE="your_passphrase"
export APEX_TESTNET="true"

# Windows
set APEX_API_KEY=your_api_key
set APEX_SECRET_KEY=your_secret_key
set APEX_PASSPHRASE=your_passphrase
set APEX_TESTNET=true
```

#### 方式B: 交互式输入

直接运行脚本，会提示输入API密钥。

### 3. 启动交易

```bash
python run_strategy.py
```

按照提示选择：
- 运行模式（安全/激进/自定义）
- 是否模拟运行
- 确认配置

**首次使用强烈建议：**
- ✅ 使用测试网（testnet=true）
- ✅ 使用模拟模式（dry_run=true）
- ✅ 选择安全保守模式

### 4. 监控运行

程序会自动：
1. 同步现有持仓
2. 设置止损止盈
3. 定期扫描市场
4. 自动执行交易
5. 生成交易报告

## 常用操作

### 查看当前持仓和风险

```bash
python -c "
from strategy_example import example_risk_check_only
example_risk_check_only()
"
```

### 单次扫描（不持续运行）

```bash
python -c "
from strategy_example import example_single_scan
example_single_scan()
"
```

### 停止程序

按 `Ctrl+C` 优雅退出，程序会：
1. 保存当前状态
2. 生成最终报告
3. 安全退出

**注意：** 停止程序不会自动平仓，现有持仓会继续保留。

## 配置自定义策略

### 修改交易对

```python
config.strategy.symbols = ["BTC-USDC", "ETH-USDC"]
```

### 调整风险参数

```python
config.risk.max_position_size = 0.05  # 最大5%仓位
config.risk.stop_loss_pct = 0.015     # 1.5%止损
config.risk.max_daily_loss_pct = 0.03 # 每日最大3%亏损
```

### 修改时间框架

```python
config.strategy.timeframe = "1h"  # 使用1小时K线
config.scan_interval = 300        # 每5分钟扫描一次
```

## 安全检查清单

开始实盘交易前，确保：

- [ ] 已在测试网充分测试
- [ ] 已在模拟模式下运行至少24小时
- [ ] 理解所有风险参数的含义
- [ ] 设置了合理的止损和每日亏损限制
- [ ] API密钥只有交易权限，没有提现权限
- [ ] 只投入可以承受损失的资金
- [ ] 了解如何紧急停止和平仓

## 推荐配置

### 新手配置（最安全）

```python
from strategy_config import create_safe_config

config = create_safe_config()
# 特点：
# - 最大2倍杠杆
# - 最大5%仓位
# - 1.5%止损
# - 使用1小时K线
```

### 进阶配置（平衡）

```python
config.risk.max_leverage = 3.0
config.risk.max_position_size = 0.10
config.risk.stop_loss_pct = 0.02
config.strategy.timeframe = "15m"
```

### 激进配置（高风险）

```python
from strategy_config import create_aggressive_config

config = create_aggressive_config()
# 警告：仅适合有经验的交易者
```

## 故障排除

### 问题1: 无法连接到API

**解决方案：**
- 检查API密钥是否正确
- 确认网络连接正常
- 检查是否选择了正确的网络（主网/测试网）

### 问题2: 程序报错退出

**解决方案：**
- 查看 `logs/` 目录下的日志文件
- 检查是否是网络问题
- 重启程序，状态会自动恢复

### 问题3: 没有生成交易信号

**可能原因：**
- 市场波动性过大/过小
- 成交量不足
- 信号强度未达到阈值
- 已达到每日交易次数限制

**解决方案：**
- 降低 `min_signal_strength`
- 调整技术指标参数
- 查看日志了解具体原因

### 问题4: 想要紧急平仓

**方案1:** 直接在Apex交易所网页端操作

**方案2:** 运行风险检查脚本
```python
from strategy_example import example_risk_check_only
# 设置 config.dry_run = False
# 手动调用 risk_manager._close_all_positions("紧急平仓")
```

## 下一步

1. **阅读完整文档：** [README_STRATEGY.md](README_STRATEGY.md)
2. **了解配置选项：** [strategy_config.py](strategy_config.py)
3. **查看示例代码：** [strategy_example.py](strategy_example.py)
4. **监控日志：** `logs/trading_*.log`

## 获取帮助

如遇到问题：
1. 查看日志文件 `logs/trading_*.log`
2. 检查是否有错误或警告信息
3. 参考 [README_STRATEGY.md](README_STRATEGY.md) 中的常见问题

---

**免责声明：** 交易有风险，投资需谨慎。请只使用你能承受损失的资金。
