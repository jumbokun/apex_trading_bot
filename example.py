"""
Apex Exchange SDK 使用示例
"""
from apex_trading_bot import ApexExchange, ApexAPIError


def example_public_api():
    """公共 API 示例（无需认证）"""
    print("=" * 50)
    print("公共 API 示例")
    print("=" * 50)
    
    # 公共 API 不需要真实的 API 密钥
    apex = ApexExchange(
        api_key="",
        secret_key="",
        passphrase="",
        testnet=True  # 使用测试网
    )
    
    try:
        with apex:
            # 获取系统时间
            print("\n1. 获取系统时间:")
            time_data = apex.public.get_system_time()
            print(time_data)
            
            # 获取配置数据
            print("\n2. 获取配置数据:")
            config = apex.public.get_all_config_data()
            print(f"配置数据已获取，包含 {len(config.get('data', {}).get('symbols', []))} 个交易对")
            
            # 获取市场深度
            print("\n3. 获取市场深度 (BTC-USDC):")
            depth = apex.public.get_market_depth("BTC-USDC", limit=5)
            print(depth)
            
            # 获取 K 线数据
            print("\n4. 获取 K 线数据 (BTC-USDC, 1小时):")
            klines = apex.public.get_candlestick_data(
                symbol="BTC-USDC",
                interval="1h",
                limit=10
            )
            print(f"获取到 {len(klines.get('data', []))} 根 K 线")
            
            # 获取 Ticker
            print("\n5. 获取 Ticker (BTC-USDC):")
            ticker = apex.public.get_ticker_data("BTC-USDC")
            print(ticker)
            
    except ApexAPIError as e:
        print(f"API 错误: {e.code} - {e.message}")
    except Exception as e:
        print(f"错误: {e}")


def example_account_api():
    """账户 API 示例"""
    print("\n" + "=" * 50)
    print("账户 API 示例")
    print("=" * 50)
    
    # 请替换为您的真实 API 密钥
    apex = ApexExchange(
        api_key="your_api_key",
        secret_key="your_secret_key",
        passphrase="your_passphrase",
        testnet=True
    )
    
    try:
        with apex:
            # 获取账户数据
            print("\n1. 获取账户数据:")
            account = apex.account.get_account_data()
            print(account)
            
            # 获取账户余额
            print("\n2. 获取账户余额:")
            balance = apex.account.get_account_balance()
            print(balance)
            
            # 获取资金费率
            print("\n3. 获取资金费率:")
            funding_rate = apex.account.get_funding_rate()
            print(funding_rate)
            
    except ApexAPIError as e:
        print(f"API 错误: {e.code} - {e.message}")
    except Exception as e:
        print(f"错误: {e}")


def example_trade_api():
    """交易 API 示例"""
    print("\n" + "=" * 50)
    print("交易 API 示例")
    print("=" * 50)
    
    # 请替换为您的真实 API 密钥
    apex = ApexExchange(
        api_key="your_api_key",
        secret_key="your_secret_key",
        passphrase="your_passphrase",
        testnet=True
    )
    
    try:
        with apex:
            # 查询当前挂单
            print("\n1. 查询当前挂单:")
            open_orders = apex.trade.get_open_orders()
            print(f"当前有 {len(open_orders.get('data', {}).get('list', []))} 个挂单")
            
            # 查询订单历史
            print("\n2. 查询订单历史:")
            order_history = apex.trade.get_order_history(
                page=1,
                limit=10
            )
            print(f"获取到 {len(order_history.get('data', {}).get('list', []))} 条订单记录")
            
            # 查询成交历史
            print("\n3. 查询成交历史:")
            trade_history = apex.trade.get_trade_history(
                page=1,
                limit=10
            )
            print(f"获取到 {len(trade_history.get('data', {}).get('list', []))} 条成交记录")
            
            # 注意：实际下单需要确保账户有足够的余额
            # 以下代码仅作为示例，请谨慎使用
            
            # print("\n4. 创建限价买单示例（已注释，避免误操作）:")
            # order = apex.trade.create_order(
            #     symbol="BTC-USDC",
            #     side="BUY",
            #     type="LIMIT",
            #     size="0.001",
            #     price="30000",
            #     time_in_force="GTC",
            #     client_order_id="test_order_001"
            # )
            # print(order)
            
    except ApexAPIError as e:
        print(f"API 错误: {e.code} - {e.message}")
    except Exception as e:
        print(f"错误: {e}")


def example_asset_api():
    """资产管理 API 示例"""
    print("\n" + "=" * 50)
    print("资产管理 API 示例")
    print("=" * 50)
    
    # 请替换为您的真实 API 密钥
    apex = ApexExchange(
        api_key="your_api_key",
        secret_key="your_secret_key",
        passphrase="your_passphrase",
        testnet=True
    )
    
    try:
        with apex:
            # 获取转账限额
            print("\n1. 获取合约账户转账限额:")
            limits = apex.asset.get_contract_account_transfer_limits()
            print(limits)
            
            # 获取提现手续费
            print("\n2. 获取提现手续费:")
            fees = apex.asset.get_withdrawal_fees()
            print(fees)
            
            # 获取充值和提现数据
            print("\n3. 获取充值和提现数据:")
            transfers = apex.asset.get_deposit_withdraw_data(
                page=1,
                limit=10
            )
            print(f"获取到 {len(transfers.get('data', {}).get('list', []))} 条记录")
            
            # 注意：实际转账和提现需要确保账户有足够的余额
            # 以下代码仅作为示例，请谨慎使用
            
            # print("\n4. 转账示例（已注释，避免误操作）:")
            # transfer = apex.asset.transfer_fund_to_contract(
            #     currency_id="USDC",
            #     amount="100"
            # )
            # print(transfer)
            
    except ApexAPIError as e:
        print(f"API 错误: {e.code} - {e.message}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    # 运行公共 API 示例（无需认证）
    example_public_api()
    
    # 以下示例需要真实的 API 密钥，请取消注释并填入您的密钥
    # example_account_api()
    # example_trade_api()
    # example_asset_api()
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("=" * 50)
    print("\n提示：")
    print("1. 公共 API 示例可以直接运行，无需 API 密钥")
    print("2. 其他示例需要真实的 API 密钥，请在代码中填入")
    print("3. 建议先在测试网上测试")

