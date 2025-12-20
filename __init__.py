"""
Apex Exchange Python SDK
========================
Apex Exchange API v3 的 Python SDK
"""
from .apex_client import ApexClient, ApexAPIError
from .user import UserAPI
from .account import AccountAPI
from .asset import AssetAPI
from .trade import TradeAPI
from .public import PublicAPI


class ApexExchange:
    """Apex Exchange 主类，整合所有 API 模块"""
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: str = None,
        testnet: bool = False,
    ):
        """
        初始化 Apex Exchange 客户端
        
        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            base_url: 自定义基础 URL（可选）
            testnet: 是否使用测试网
        """
        self.client = ApexClient(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            base_url=base_url,
            testnet=testnet,
        )
        
        # 初始化各个 API 模块
        self.user = UserAPI(self.client)
        self.account = AccountAPI(self.client)
        self.asset = AssetAPI(self.client)
        self.trade = TradeAPI(self.client)
        self.public = PublicAPI(self.client)
    
    def __enter__(self):
        self.client.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.__exit__(exc_type, exc_val, exc_tb)


__all__ = [
    "ApexExchange",
    "ApexClient",
    "ApexAPIError",
    "UserAPI",
    "AccountAPI",
    "AssetAPI",
    "TradeAPI",
    "PublicAPI",
]

