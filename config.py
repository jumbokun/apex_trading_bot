"""
配置文件
支持从环境变量读取配置
"""
import os
from typing import Optional

# 尝试加载 .env 文件（如果安装了 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装 python-dotenv，则跳过
    pass


class ApexConfig:
    """Apex Exchange 配置类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        base_url: Optional[str] = None,
        testnet: Optional[bool] = None,
    ):
        """
        初始化配置
        
        Args:
            api_key: API Key（如果为 None，则从环境变量读取）
            secret_key: Secret Key（如果为 None，则从环境变量读取）
            passphrase: Passphrase（如果为 None，则从环境变量读取）
            base_url: 自定义基础 URL（可选）
            testnet: 是否使用测试网（如果为 None，则从环境变量读取）
        """
        self.api_key = api_key or os.getenv("APEX_API_KEY", "")
        self.secret_key = secret_key or os.getenv("APEX_SECRET_KEY", "")
        self.passphrase = passphrase or os.getenv("APEX_PASSPHRASE", "")
        
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = os.getenv("APEX_BASE_URL")
        
        if testnet is not None:
            self.testnet = testnet
        else:
            testnet_str = os.getenv("APEX_TESTNET", "false").lower()
            self.testnet = testnet_str in ("true", "1", "yes")

