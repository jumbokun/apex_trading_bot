"""
用户相关 API
"""
from typing import Optional, Dict
from .apex_client import ApexClient


class UserAPI:
    """用户相关 API"""
    
    def __init__(self, client: ApexClient):
        self.client = client
    
    def generate_nonce(
        self,
        refresh: str = "false",
        l2_key: Optional[str] = None,
        eth_address: Optional[str] = None,
        chain_id: Optional[int] = None
    ) -> Dict:
        """
        生成 nonce
        POST /v3/user/generate-nonce
        
        Args:
            refresh: 是否刷新 nonce
            l2_key: L2 密钥
            eth_address: 以太坊地址
            chain_id: 链 ID
        
        Returns:
            包含 nonce 的响应数据
        """
        data = {"refresh": refresh}
        if l2_key:
            data["l2Key"] = l2_key
        if eth_address:
            data["ethAddress"] = eth_address
        if chain_id:
            data["chainId"] = str(chain_id)
        
        return self.client.post("/v3/user/generate-nonce", data)
    
    def register_user(
        self,
        nonce: str,
        l2_key: str,
        seeds: str,
        ethereum_address: str
    ) -> Dict:
        """
        注册用户
        POST /v3/user/registration
        
        Args:
            nonce: Nonce
            l2_key: L2 密钥
            seeds: Seeds
            ethereum_address: 以太坊地址
        
        Returns:
            注册结果，包含 apiKey 信息
        """
        data = {
            "nonce": nonce,
            "l2Key": l2_key,
            "seeds": seeds,
            "ethereumAddress": ethereum_address,
        }
        return self.client.post("/v3/user/registration", data)
    
    def get_user_data(self) -> Dict:
        """
        获取用户数据
        GET /v3/user
        
        Returns:
            用户数据
        """
        return self.client.get("/v3/user")
    
    def edit_user_data(
        self,
        username: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        编辑用户数据
        POST /v3/user
        
        Args:
            username: 用户名
            email: 邮箱
            **kwargs: 其他可编辑字段
        
        Returns:
            更新后的用户数据
        """
        data = {}
        if username:
            data["username"] = username
        if email:
            data["email"] = email
        data.update(kwargs)
        
        return self.client.post("/v3/user", data)

