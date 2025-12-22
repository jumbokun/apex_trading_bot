# Apex Trading Bot

FOMO策略自动交易机器人，基于 Apex Exchange (omni.apex.exchange) 实现。

## 策略概述

**FOMO (Fear Of Missing Out) 突破策略**

核心逻辑：
1. **入场**: 价格突破60根K线高点 + 成交量2倍确认
2. **止损**: 1.5倍ATR止损
3. **止盈**: 盈利达3R后启用7倍ATR追踪止损
4. **方向**: 默认只做多（回测显示多头表现更优）

回测表现（14天，6币种）：
- 胜率: ~45%
- 预期盈利: +815U (初始资金5000U)

## 文件结构

```
apex_trading_bot/
├── fomo_strategy.py      # 策略核心逻辑
├── live_trader_v2.py     # 实盘交易器（使用apexomni SDK）
├── run_backtest.py       # 回测工具
├── optimize_params.py    # 参数优化工具
├── .env.example          # 配置模板
├── requirements.txt      # 依赖
└── LICENSE               # MIT License
```

## 快速开始

### 1. 安装依赖

```bash
pip install apexomni numpy
```

### 2. 配置API密钥

复制 `.env.example` 为 `.env`，填入你的 Apex Exchange API 配置：

```env
APEX_API_KEY=your_api_key
APEX_SECRET_KEY=your_secret_key
APEX_PASSPHRASE=your_passphrase
APEX_TESTNET=false
OMNIKEY=your_omnikey_from_apex_website
```

**获取 OMNIKEY**:
1. 登录 https://omni.apex.exchange
2. 进入 API 管理页面
3. 复制 OMNIKEY（用于签名交易）

### 3. 运行回测

```bash
python run_backtest.py
```

### 4. 运行实盘

```bash
# 使用 Python 3.12（推荐）
py -3.12 live_trader_v2.py
```

启动后会提示确认，输入 `YES` 开始交易。

## 监控交易

### 查看日志

**Windows PowerShell:**
```powershell
Get-Content "trading.log" -Wait -Tail 50
```

**Linux/Mac:**
```bash
tail -f trading.log
```

### 日志位置
日志文件 `trading.log` 保存在项目目录下。

## 交易对

默认监控46个高流动性交易对（日均交易量 > $5M），包括：

- **主流币**: BTC, ETH, SOL
- **大盘币**: AAVE, LINK, UNI, NEAR, APT, ARB, OP...
- **Meme币**: PEPE, BONK, WIF, PENGU, FARTCOIN...
- **DeFi**: PENDLE, CRV, LDO, JUP, ENA...
- **热门新币**: HYPE, VIRTUAL, IP, KAITO, LAYER...

## 风险控制

- 每笔交易最大风险: 50U
- 每日最大亏损: 150U
- 杠杆范围: 2-5x（根据波动率自动调整）
- 单一品种冷却期

## 参数优化

运行参数优化工具找到最佳设置：

```bash
python optimize_params.py
```

当前最优参数：
- `breakout_lookback`: 60
- `vol_ratio_entry`: 2.0
- `k_stop_atr`: 1.5
- `trail_k_atr`: 7.0
- `enable_trail_after_R`: 3.0

## 注意事项

1. **风险提示**: 加密货币交易有高风险，请只用可承受损失的资金
2. **网络要求**: 需要稳定的网络连接访问 Apex Exchange API
3. **API限制**: 遵守交易所API调用频率限制
4. **测试建议**: 先在测试网验证，再使用主网

## License

MIT License
