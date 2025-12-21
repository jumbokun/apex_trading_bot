# 项目结构说明

## 📁 文件组织

```
apex_trading_bot/
├── 📚 核心SDK文件
│   ├── __init__.py              # SDK入口
│   ├── apex_client.py           # API客户端（HTTP请求、签名）
│   ├── account.py               # 账户API
│   ├── trade.py                 # 交易API
│   ├── asset.py                 # 资产API
│   ├── user.py                  # 用户API
│   ├── public.py                # 公共API
│   └── config.py                # SDK配置
│
├── 🤖 交易策略系统（新增）
│   ├── strategy_config.py       # 策略配置定义
│   ├── market_analyzer.py       # 市场分析和信号生成
│   ├── position_manager.py      # 仓位管理
│   ├── risk_manager.py          # 风险管理和止损止盈
│   └── trading_engine.py        # 交易引擎主控制器
│
├── 📖 文档
│   ├── README.md                # SDK基础文档
│   ├── README_STRATEGY.md       # 策略系统完整文档
│   ├── QUICKSTART.md            # 5分钟快速开始
│   ├── FEATURES.md              # 功能特性详解
│   └── PROJECT_STRUCTURE.md     # 本文件
│
├── 🚀 可执行脚本
│   ├── run_strategy.py          # 快速启动脚本（交互式）
│   ├── strategy_example.py      # 策略使用示例
│   ├── test_strategy.py         # 系统测试脚本
│   └── example.py               # SDK基础示例
│
├── ⚙️ 配置文件
│   ├── config.yaml.example      # 配置文件示例
│   ├── requirements.txt         # SDK依赖
│   └── requirements_strategy.txt # 策略系统依赖
│
└── 📋 其他
    └── setup.py                 # 安装配置
```

---

## 📚 核心模块说明

### 1. SDK基础模块

#### `apex_client.py` - API客户端
- HTTP请求封装
- HMAC-SHA256签名生成
- 错误处理
- 公共/私有API区分

#### `account.py` - 账户管理
- 获取账户数据和持仓
- 查询账户余额
- 获取资金费率
- 历史盈亏查询

#### `trade.py` - 交易操作
- 创建订单（限价/市价）
- 取消订单
- 查询挂单和历史
- 获取成交记录

#### `asset.py` - 资产管理
- 账户转账
- 充值提现
- 查询手续费
- 转账限额查询

#### `public.py` - 公共数据
- 市场深度
- K线数据
- Ticker行情
- 系统时间
- 配置信息

---

### 2. 交易策略系统

#### `strategy_config.py` - 策略配置 🎯
**作用：** 定义所有策略和风险参数

**主要类：**
- `RiskConfig`: 风险控制配置
  - 仓位限制
  - 止损止盈
  - 爆仓保护
  - 每日限制
  - 市场过滤

- `StrategyConfig`: 交易策略配置
  - 交易对选择
  - 技术指标参数
  - 信号强度要求
  - 仓位管理规则

- `TradingConfig`: 总体配置
  - API配置
  - 运行模式
  - 扫描间隔
  - 状态持久化

**预设配置：**
- `create_safe_config()`: 安全保守配置
- `create_aggressive_config()`: 激进配置
- `load_default_config()`: 默认配置

---

#### `market_analyzer.py` - 市场分析 📊
**作用：** 扫描市场、计算指标、生成交易信号

**主要类：**

1. `TechnicalIndicators` - 技术指标计算器
   - `calculate_sma()`: 简单移动平均
   - `calculate_ema()`: 指数移动平均
   - `calculate_rsi()`: 相对强弱指标
   - `calculate_volatility()`: 波动率
   - `calculate_atr()`: 平均真实波动范围

2. `MarketSignal` - 市场信号数据类
   - 信号类型（BUY/SELL/HOLD/CLOSE）
   - 信号强度（0-1）
   - 价格和时间戳
   - 信号原因说明

3. `MarketAnalyzer` - 市场分析器
   - `scan_market()`: 扫描所有交易对
   - `_analyze_symbol()`: 分析单个交易对
   - `_generate_signal()`: 生成交易信号
   - `_is_market_tradable()`: 检查市场条件
   - `check_exit_signal()`: 检查平仓信号

---

#### `position_manager.py` - 仓位管理 💼
**作用：** 管理持仓、同步状态、风险限制检查

**主要类：**

1. `Position` - 持仓数据类
   - 基本信息（交易对、方向、数量、价格）
   - 盈亏计算（未实现盈亏、百分比）
   - 止损止盈价格
   - 追踪止损管理
   - 保证金和杠杆信息

   **关键方法：**
   - `update_price()`: 更新价格和盈亏
   - `should_stop_loss()`: 检查止损触发
   - `should_take_profit()`: 检查止盈触发
   - `update_trailing_stop()`: 更新追踪止损

2. `PositionManager` - 仓位管理器
   - `sync_positions()`: 从交易所同步持仓
   - `check_risk_limits()`: 检查所有风险限制
   - `can_open_new_position()`: 检查是否可开仓
   - `record_trade()`: 记录交易统计
   - `save_state()` / `load_state()`: 状态持久化

---

#### `risk_manager.py` - 风险管理 🛡️
**作用：** 执行风险控制、止损止盈、爆仓保护

**主要类：**

`RiskManager` - 风险管理器

**核心功能：**

1. **持仓监控**
   - `monitor_positions()`: 监控所有持仓风险
   - 自动执行止损止盈
   - 爆仓保护触发
   - 追踪止损更新

2. **平仓执行**
   - `_close_position()`: 平仓单个持仓
   - `_close_all_positions()`: 平掉所有持仓
   - 支持模拟和实盘模式

3. **仓位计算**
   - `calculate_position_size()`: 计算合适的仓位大小
   - `check_position_size()`: 检查仓位是否符合限制
   - 基于信号强度动态调整

4. **加减仓判断**
   - `should_scale_in()`: 是否应该加仓
   - `should_scale_out()`: 是否应该减仓

5. **风险报告**
   - `get_risk_report()`: 生成详细风险报告
   - 账户状态
   - 持仓详情
   - 盈亏统计

---

#### `trading_engine.py` - 交易引擎 🚀
**作用：** 主控制器，协调所有模块，执行交易循环

**主要类：**

`TradingEngine` - 交易引擎

**核心流程：**

1. **初始化**
   - 创建市场分析器
   - 创建仓位管理器
   - 创建风险管理器
   - 加载历史状态

2. **启动检查**
   - 同步现有持仓
   - 设置止损止盈
   - 执行初始风险检查

3. **主循环** (每次迭代)
   ```
   1. 监控风险
      ├── 检查止损止盈
      ├── 检查保证金率
      ├── 检查每日限制
      └── 执行必要的平仓

   2. 扫描市场
      ├── 获取K线数据
      ├── 计算技术指标
      ├── 生成交易信号
      └── 过滤信号强度

   3. 处理信号
      ├── 检查是否已有持仓
      ├── 决定开仓/加仓/减仓/平仓
      ├── 计算仓位大小
      └── 执行订单

   4. 生成报告
      ├── 账户状态
      ├── 持仓详情
      ├── 今日统计
      └── 自动评价

   5. 保存状态

   6. 等待下次扫描
   ```

4. **信号处理**
   - `_handle_signal()`: 处理交易信号
   - `_handle_signal_with_position()`: 已有持仓时的处理
   - `_handle_signal_without_position()`: 无持仓时的处理
   - `_open_position()`: 执行开仓

5. **报告和评价**
   - `_generate_report()`: 生成交易报告
   - `_auto_evaluate()`: 自动评价系统
   - 实时输出关键指标

---

## 🚀 可执行脚本

### `run_strategy.py` - 快速启动 ⭐
**推荐使用**

**特点：**
- 交互式配置
- 环境变量支持
- 自动日志设置
- 连接测试
- 安全确认

**使用方法：**
```bash
python run_strategy.py
```

---

### `strategy_example.py` - 示例代码
包含4个使用示例：
1. `example_safe_trading()`: 安全保守策略
2. `example_custom_strategy()`: 自定义策略
3. `example_single_scan()`: 单次扫描
4. `example_risk_check_only()`: 仅风险检查

---

### `test_strategy.py` - 系统测试
运行5个测试用例：
1. 技术指标计算测试
2. 仓位管理测试
3. 市场分析器测试（需要网络）
4. 风险计算测试
5. 配置预设测试

**使用方法：**
```bash
python test_strategy.py
```

---

## 📖 文档

### `README.md` - SDK文档
- API使用说明
- 认证方式
- 基础示例

### `README_STRATEGY.md` - 策略系统文档 ⭐
**最重要的文档**
- 核心特性详解
- 安装配置
- 使用场景
- 配置参数说明
- 安全保障机制
- 常见问题

### `QUICKSTART.md` - 快速开始
- 5分钟入门
- 常用操作
- 安全检查清单
- 故障排除

### `FEATURES.md` - 功能特性
- 详细功能列表
- 为什么安全
- 配置对比
- 使用场景
- 技术优势

---

## ⚙️ 配置文件

### `config.yaml.example` - 配置模板
YAML格式的配置文件示例，包含：
- API配置
- 运行模式
- 策略参数
- 风险参数
- 预设说明

### `requirements_strategy.txt` - 依赖列表
```
numpy>=1.24.0
loguru>=0.7.0
httpx>=0.25.0
```

---

## 🎯 使用建议

### 新手路径
1. 阅读 `QUICKSTART.md`
2. 运行 `python test_strategy.py` 测试系统
3. 使用 `python run_strategy.py` 启动（选择安全模式）
4. 开始时使用测试网 + 模拟模式
5. 观察日志和报告
6. 逐步调整参数

### 进阶路径
1. 阅读 `README_STRATEGY.md` 了解所有功能
2. 阅读 `FEATURES.md` 了解实现细节
3. 查看 `strategy_example.py` 学习自定义
4. 修改 `strategy_config.py` 创建自己的策略
5. 实盘测试（小资金）

### 开发者路径
1. 理解项目结构（本文档）
2. 阅读各模块源代码
3. 修改或扩展功能：
   - 添加新的技术指标 → `market_analyzer.py`
   - 修改信号生成逻辑 → `MarketAnalyzer._generate_signal()`
   - 添加新的风险控制 → `risk_manager.py`
   - 改进仓位管理 → `position_manager.py`

---

## 📊 数据流图

```
启动
  ↓
初始化各模块
  ↓
同步现有持仓 ← PositionManager
  ↓
设置止损止盈 ← RiskManager
  ↓
┌─────────── 主循环 ───────────┐
│                              │
│  1. 监控风险 ← RiskManager   │
│     ├── 检查止损止盈         │
│     ├── 检查保证金率         │
│     └── 执行平仓             │
│                              │
│  2. 扫描市场 ← MarketAnalyzer│
│     ├── 获取K线数据          │
│     ├── 计算技术指标         │
│     └── 生成信号             │
│                              │
│  3. 处理信号 ← TradingEngine │
│     ├── 检查持仓状态         │
│     ├── 计算仓位             │
│     └── 执行订单             │
│                              │
│  4. 生成报告                 │
│     └── 自动评价             │
│                              │
│  5. 保存状态                 │
│                              │
│  6. 等待间隔                 │
│     ↓                        │
└──────────────────────────────┘
         ↓ (Ctrl+C)
     优雅退出
```

---

## 🔧 扩展指南

### 添加新的技术指标

编辑 `market_analyzer.py`:

```python
class TechnicalIndicators:
    @staticmethod
    def calculate_new_indicator(data, period):
        # 实现你的指标
        return result
```

### 添加新的交易策略

编辑 `market_analyzer.py` 的 `_generate_signal()` 方法：

```python
def _generate_signal(self, symbol, current_price, indicators):
    # 添加你的策略逻辑
    if your_condition:
        buy_score += 0.3
    # ...
```

### 添加新的风险控制

编辑 `risk_manager.py`:

```python
class RiskManager:
    def custom_risk_check(self):
        # 实现你的风险控制
        pass
```

---

## 📝 日志位置

- 控制台输出：实时显示
- 文件日志：`logs/trading_YYYY-MM-DD.log`
- 状态文件：`trading_state.json`（可配置）

---

## 🤝 贡献指南

欢迎贡献改进：
1. 优化技术指标计算
2. 添加新的交易策略
3. 改进风险控制机制
4. 完善文档和示例
5. 修复bug

---

## ⚠️ 重要提醒

1. 先在测试网测试
2. 使用模拟模式验证策略
3. 只投入可承受损失的资金
4. 定期检查持仓和日志
5. 保管好API密钥

---

祝交易顺利！🚀
