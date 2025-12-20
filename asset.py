"""
资产相关 API（转账、提现、充值）
"""
from typing import Optional, Dict
from .apex_client import ApexClient


class AssetAPI:
    """资产相关 API"""
    
    def __init__(self, client: ApexClient):
        self.client = client
    
    def transfer_fund_to_contract(
        self,
        currency_id: str,
        amount: str
    ) -> Dict:
        """
        从资金账户转账到合约账户
        POST /v3/asset/transfer-in
        
        Args:
            currency_id: 币种 ID（如 USDC, USDT）
            amount: 转账金额
        
        Returns:
            转账结果
        """
        data = {
            "currencyId": currency_id,
            "amount": amount,
        }
        return self.client.post("/v3/asset/transfer-in", data)
    
    def transfer_contract_to_fund(
        self,
        currency_id: str,
        amount: str
    ) -> Dict:
        """
        从合约账户转账到资金账户
        POST /v3/asset/transfer-out
        
        Args:
            currency_id: 币种 ID
            amount: 转账金额
        
        Returns:
            转账结果
        """
        data = {
            "currencyId": currency_id,
            "amount": amount,
        }
        return self.client.post("/v3/asset/transfer-out", data)
    
    def withdraw(
        self,
        currency_id: str,
        amount: str,
        chain_id: int,
        address: str,
        client_withdraw_id: Optional[str] = None
    ) -> Dict:
        """
        提现
        POST /v3/asset/withdraw
        
        Args:
            currency_id: 币种 ID
            amount: 提现金额
            chain_id: 链 ID
            address: 提现地址
            client_withdraw_id: 客户端提现 ID（可选）
        
        Returns:
            提现结果
        """
        data = {
            "currencyId": currency_id,
            "amount": amount,
            "chainId": str(chain_id),
            "address": address,
        }
        if client_withdraw_id:
            data["clientWithdrawId"] = client_withdraw_id
        
        return self.client.post("/v3/asset/withdraw", data)
    
    def get_withdrawal_fees(
        self,
        currency_id: Optional[str] = None,
        chain_id: Optional[int] = None
    ) -> Dict:
        """
        获取提现手续费
        GET /v3/asset/withdrawal-fees
        
        Args:
            currency_id: 币种 ID（可选）
            chain_id: 链 ID（可选）
        
        Returns:
            提现手续费信息
        """
        params = {}
        if currency_id:
            params["currencyId"] = currency_id
        if chain_id:
            params["chainId"] = str(chain_id)
        
        return self.client.get("/v3/asset/withdrawal-fees", params)
    
    def submit_withdraw_claim(
        self,
        withdraw_id: str
    ) -> Dict:
        """
        提交提现声明
        POST /v3/asset/submit-withdraw-claim
        
        Args:
            withdraw_id: 提现 ID
        
        Returns:
            提交结果
        """
        data = {"withdrawId": withdraw_id}
        return self.client.post("/v3/asset/submit-withdraw-claim", data)
    
    def get_contract_account_transfer_limits(
        self,
        currency_id: Optional[str] = None
    ) -> Dict:
        """
        获取合约账户转账限额
        GET /v3/asset/contract-account-transfer-limits
        
        Args:
            currency_id: 币种 ID（可选）
        
        Returns:
            转账限额信息
        """
        params = {}
        if currency_id:
            params["currencyId"] = currency_id
        
        return self.client.get("/v3/asset/contract-account-transfer-limits", params)
    
    def get_repayment_price(
        self,
        symbol: str
    ) -> Dict:
        """
        获取还款价格
        GET /v3/asset/repayment-price
        
        Args:
            symbol: 交易对符号
        
        Returns:
            还款价格信息
        """
        params = {"symbol": symbol}
        return self.client.get("/v3/asset/repayment-price", params)
    
    def manual_repayment_v3(
        self,
        symbol: str,
        amount: str
    ) -> Dict:
        """
        手动还款 V3
        POST /v3/asset/manual-repayment-v3
        
        Args:
            symbol: 交易对符号
            amount: 还款金额
        
        Returns:
            还款结果
        """
        data = {
            "symbol": symbol,
            "amount": amount,
        }
        return self.client.post("/v3/asset/manual-repayment-v3", data)
    
    def get_deposit_withdraw_data(
        self,
        currency_id: Optional[str] = None,
        chain_ids: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict:
        """
        获取充值和提现数据
        GET /v3/transfers
        
        Args:
            currency_id: 币种 ID（可选）
            chain_ids: 链 ID 列表，逗号分隔（可选）
            page: 页码
            limit: 每页数量
            start_time: 开始时间（Unix 时间戳）
            end_time: 结束时间（Unix 时间戳）
        
        Returns:
            充值和提现数据
        """
        params = {}
        if currency_id:
            params["currencyId"] = currency_id
        if chain_ids:
            params["chainIds"] = chain_ids
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        
        return self.client.get("/v3/transfers", params)

