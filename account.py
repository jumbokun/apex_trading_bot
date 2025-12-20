"""
账户相关 API
"""
from typing import Optional, Dict
from .apex_client import ApexClient


class AccountAPI:
    """账户相关 API"""
    
    def __init__(self, client: ApexClient):
        self.client = client
    
    def get_account_data(self) -> Dict:
        """
        获取账户数据和持仓
        GET /v3/account
        
        Returns:
            账户数据，包含持仓信息
        """
        return self.client.get("/v3/account")
    
    def get_account_balance(
        self,
        currency_id: Optional[str] = None
    ) -> Dict:
        """
        获取账户余额
        GET /v3/account/balance
        
        Args:
            currency_id: 币种 ID（可选，如 USDC）
        
        Returns:
            账户余额信息
        """
        params = {}
        if currency_id:
            params["currencyId"] = currency_id
        
        return self.client.get("/v3/account/balance", params)
    
    def get_funding_rate(
        self,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        获取资金费率
        GET /v3/account/funding-rate
        
        Args:
            symbol: 交易对符号（可选）
        
        Returns:
            资金费率信息
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        return self.client.get("/v3/account/funding-rate", params)
    
    def get_historical_pnl(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        获取历史盈亏
        GET /v3/account/historical-pnl
        
        Args:
            start_time: 开始时间（Unix 时间戳）
            end_time: 结束时间（Unix 时间戳）
            page: 页码
            limit: 每页数量
        
        Returns:
            历史盈亏数据
        """
        params = {}
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        
        return self.client.get("/v3/account/historical-pnl", params)
    
    def get_yesterday_pnl(self) -> Dict:
        """
        获取昨日盈亏
        GET /v3/account/yesterday-pnl
        
        Returns:
            昨日盈亏数据
        """
        return self.client.get("/v3/account/yesterday-pnl")
    
    def get_historical_asset_value(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        获取历史资产价值
        GET /v3/account/historical-asset-value
        
        Args:
            start_time: 开始时间（Unix 时间戳）
            end_time: 结束时间（Unix 时间戳）
            page: 页码
            limit: 每页数量
        
        Returns:
            历史资产价值数据
        """
        params = {}
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        
        return self.client.get("/v3/account/historical-asset-value", params)
    
    def set_initial_margin_rate(
        self,
        symbol: str,
        initial_margin_rate: str
    ) -> Dict:
        """
        设置合约的初始保证金率
        POST /v3/account/set-initial-margin-rate
        
        Args:
            symbol: 交易对符号
            initial_margin_rate: 初始保证金率
        
        Returns:
            设置结果
        """
        data = {
            "symbol": symbol,
            "initialMarginRate": initial_margin_rate,
        }
        return self.client.post("/v3/account/set-initial-margin-rate", data)

