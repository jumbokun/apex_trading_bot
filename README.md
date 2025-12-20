# Apex Exchange Python SDK

Apex Exchange API v3 的 Python SDK，提供完整的交易、账户、资产管理和市场数据功能。

## 功能特性

- ✅ 完整的 API v3 支持
- ✅ 用户管理（注册、获取用户数据）
- ✅ 账户管理（余额查询、持仓、盈亏历史）
- ✅ 资产管理（转账、提现、充值查询）
- ✅ 交易功能（下单、取消订单、查询订单、成交历史）
- ✅ 公共市场数据（K线、深度、Ticker、资金费率）
- ✅ 自动签名认证
- ✅ 支持主网和测试网
- ✅ 完善的错误处理

## 安装

```bash
pip install -r requirements.txt
```

或者使用 setup.py：

```bash
pip install .
```

## 配置

### 使用环境变量文件（推荐）

1. 复制示例配置文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入您的真实 API 密钥：
```env
APEX_API_KEY=your_api_key_here
APEX_SECRET_KEY=your_secret_key_here
APEX_PASSPHRASE=your_passphrase_here
APEX_TESTNET=true
```

3. `.env` 文件已被 `.gitignore` 忽略，不会提交到代码仓库

### 直接传入参数

您也可以在代码中直接传入 API 密钥（不推荐用于生产环境）：

```python
apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase",
    testnet=True
)
```

### 使用配置类

```python
from apex_trading_bot import ApexExchange
from apex_trading_bot.config import ApexConfig

# 从环境变量读取配置
config = ApexConfig()
apex = ApexExchange(
    api_key=config.api_key,
    secret_key=config.secret_key,
    passphrase=config.passphrase,
    testnet=config.testnet
)
```

## 快速开始

### 1. 初始化客户端

```python
from apex_trading_bot import ApexExchange

# 使用主网
apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase",
    testnet=False
)

# 使用测试网
apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase",
    testnet=True
)

# 使用上下文管理器（推荐）
with apex:
    # 使用 API
    pass
```

### 2. 公共 API（无需认证）

```python
from apex_trading_bot import ApexExchange

apex = ApexExchange(
    api_key="",  # 公共 API 不需要认证
    secret_key="",
    passphrase="",
    testnet=False
)

with apex:
    # 获取系统时间
    time_data = apex.public.get_system_time()
    print(time_data)
    
    # 获取所有配置数据
    config = apex.public.get_all_config_data()
    print(config)
    
    # 获取市场深度
    depth = apex.public.get_market_depth("BTC-USDC", limit=20)
    print(depth)
    
    # 获取 K 线数据
    klines = apex.public.get_candlestick_data(
        symbol="BTC-USDC",
        interval="1h",
        limit=100
    )
    print(klines)
    
    # 获取 Ticker 数据
    ticker = apex.public.get_ticker_data("BTC-USDC")
    print(ticker)
```

### 3. 账户 API

```python
from apex_trading_bot import ApexExchange

apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase"
)

with apex:
    # 获取账户数据和持仓
    account = apex.account.get_account_data()
    print(account)
    
    # 获取账户余额
    balance = apex.account.get_account_balance(currency_id="USDC")
    print(balance)
    
    # 获取资金费率
    funding_rate = apex.account.get_funding_rate(symbol="BTC-USDC")
    print(funding_rate)
    
    # 获取历史盈亏
    pnl = apex.account.get_historical_pnl(
        start_time=1609459200,  # Unix 时间戳
        end_time=1640995200,
        page=1,
        limit=100
    )
    print(pnl)
    
    # 获取昨日盈亏
    yesterday_pnl = apex.account.get_yesterday_pnl()
    print(yesterday_pnl)
```

### 4. 交易 API

```python
from apex_trading_bot import ApexExchange

apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase"
)

with apex:
    # 创建限价买单
    order = apex.trade.create_order(
        symbol="BTC-USDC",
        side="BUY",
        type="LIMIT",
        size="0.001",
        price="50000",
        time_in_force="GTC",
        client_order_id="my_order_001"
    )
    print(order)
    
    # 创建市价卖单
    market_order = apex.trade.create_order(
        symbol="BTC-USDC",
        side="SELL",
        type="MARKET",
        size="0.001"
    )
    print(market_order)
    
    # 查询当前挂单
    open_orders = apex.trade.get_open_orders(symbol="BTC-USDC")
    print(open_orders)
    
    # 取消订单
    cancel_result = apex.trade.cancel_order(
        order_id="order_id_here",
        symbol="BTC-USDC"
    )
    print(cancel_result)
    
    # 通过客户端订单 ID 取消订单
    cancel_result = apex.trade.cancel_order_by_client_id(
        client_order_id="my_order_001",
        symbol="BTC-USDC"
    )
    print(cancel_result)
    
    # 取消所有挂单
    cancel_all = apex.trade.cancel_all_orders(symbol="BTC-USDC")
    print(cancel_all)
    
    # 查询订单历史
    order_history = apex.trade.get_order_history(
        symbol="BTC-USDC",
        page=1,
        limit=100
    )
    print(order_history)
    
    # 查询成交历史
    trade_history = apex.trade.get_trade_history(
        symbol="BTC-USDC",
        page=1,
        limit=100
    )
    print(trade_history)
```

### 5. 资产管理 API

```python
from apex_trading_bot import ApexExchange

apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase"
)

with apex:
    # 从资金账户转账到合约账户
    transfer_in = apex.asset.transfer_fund_to_contract(
        currency_id="USDC",
        amount="1000"
    )
    print(transfer_in)
    
    # 从合约账户转账到资金账户
    transfer_out = apex.asset.transfer_contract_to_fund(
        currency_id="USDC",
        amount="500"
    )
    print(transfer_out)
    
    # 提现
    withdraw = apex.asset.withdraw(
        currency_id="USDC",
        amount="100",
        chain_id=1,  # 链 ID（1=以太坊主网）
        address="0x...",
        client_withdraw_id="withdraw_001"
    )
    print(withdraw)
    
    # 获取提现手续费
    fees = apex.asset.get_withdrawal_fees(
        currency_id="USDC",
        chain_id=1
    )
    print(fees)
    
    # 获取充值和提现数据
    transfers = apex.asset.get_deposit_withdraw_data(
        currency_id="USDC",
        page=1,
        limit=100
    )
    print(transfers)
```

### 6. 用户 API

```python
from apex_trading_bot import ApexExchange

apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase"
)

with apex:
    # 获取用户数据
    user_data = apex.user.get_user_data()
    print(user_data)
    
    # 编辑用户数据
    updated = apex.user.edit_user_data(
        username="new_username",
        email="new_email@example.com"
    )
    print(updated)
```

## API 模块说明

### ApexExchange 主类

主入口类，整合所有 API 模块：

- `apex.user` - 用户相关 API
- `apex.account` - 账户相关 API
- `apex.asset` - 资产管理 API
- `apex.trade` - 交易相关 API
- `apex.public` - 公共市场数据 API

### 错误处理

```python
from apex_trading_bot import ApexExchange, ApexAPIError

apex = ApexExchange(
    api_key="your_api_key",
    secret_key="your_secret_key",
    passphrase="your_passphrase"
)

try:
    with apex:
        order = apex.trade.create_order(
            symbol="BTC-USDC",
            side="BUY",
            type="LIMIT",
            size="0.001",
            price="50000"
        )
except ApexAPIError as e:
    print(f"API 错误: {e.code} - {e.message}")
    print(f"错误数据: {e.data}")
except Exception as e:
    print(f"其他错误: {e}")
```

## 注意事项

1. **API 密钥安全**：请妥善保管您的 API 密钥，不要将其提交到代码仓库
2. **测试网**：开发时建议先使用测试网（`testnet=True`）
3. **请求频率**：请注意 API 的请求频率限制
4. **时间戳**：所有时间参数使用 Unix 时间戳（秒）
5. **金额精度**：注意交易对的精度要求，避免精度错误

## 支持的 API 端点

### 用户 API
- ✅ POST `/v3/user/generate-nonce` - 生成 nonce
- ✅ POST `/v3/user/registration` - 注册用户
- ✅ GET `/v3/user` - 获取用户数据
- ✅ POST `/v3/user` - 编辑用户数据

### 账户 API
- ✅ GET `/v3/account` - 获取账户数据和持仓
- ✅ GET `/v3/account/balance` - 获取账户余额
- ✅ GET `/v3/account/funding-rate` - 获取资金费率
- ✅ GET `/v3/account/historical-pnl` - 获取历史盈亏
- ✅ GET `/v3/account/yesterday-pnl` - 获取昨日盈亏
- ✅ GET `/v3/account/historical-asset-value` - 获取历史资产价值
- ✅ POST `/v3/account/set-initial-margin-rate` - 设置初始保证金率

### 资产管理 API
- ✅ POST `/v3/asset/transfer-in` - 从资金账户转账到合约账户
- ✅ POST `/v3/asset/transfer-out` - 从合约账户转账到资金账户
- ✅ POST `/v3/asset/withdraw` - 提现
- ✅ GET `/v3/asset/withdrawal-fees` - 获取提现手续费
- ✅ POST `/v3/asset/submit-withdraw-claim` - 提交提现声明
- ✅ GET `/v3/asset/contract-account-transfer-limits` - 获取转账限额
- ✅ GET `/v3/asset/repayment-price` - 获取还款价格
- ✅ POST `/v3/asset/manual-repayment-v3` - 手动还款
- ✅ GET `/v3/transfers` - 获取充值和提现数据

### 交易 API
- ✅ POST `/v3/order` - 创建订单
- ✅ POST `/v3/order/cancel` - 取消订单
- ✅ POST `/v3/order/cancel-by-client-order-id` - 通过客户端订单 ID 取消
- ✅ GET `/v3/order/open` - 获取当前挂单
- ✅ POST `/v3/order/cancel-all` - 取消所有挂单
- ✅ GET `/v3/order/history` - 获取订单历史
- ✅ GET `/v3/order` - 通过订单 ID 查询订单
- ✅ GET `/v3/order/by-client-order-id` - 通过客户端订单 ID 查询
- ✅ GET `/v3/trade` - 获取成交历史
- ✅ GET `/v3/order/worst-price` - 获取最差价格

### 公共 API
- ✅ GET `/v3/public/time` - 获取系统时间
- ✅ GET `/v3/public/config` - 获取所有配置数据
- ✅ GET `/v3/public/depth` - 获取市场深度
- ✅ GET `/v3/public/trade` - 获取最新交易数据
- ✅ GET `/v3/public/kline` - 获取 K 线数据
- ✅ GET `/v3/public/ticker` - 获取 Ticker 数据
- ✅ GET `/v3/public/funding-rate-history` - 获取资金费率历史

## 开发

```bash
# 克隆仓库
git clone <repository_url>
cd apex_trading_bot

# 安装依赖
pip install -r requirements.txt

# 运行测试（如果有）
python -m pytest tests/
```

## 许可证

请查看 LICENSE 文件了解详情。

## 参考文档

- [Apex Exchange API 文档](https://api-docs.pro.apex.exchange/#introduction)
- [Apex Exchange 官网](https://apex.exchange)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0
- 初始版本
- 支持完整的 API v3 功能
- 包含用户、账户、资产、交易和公共 API 模块
