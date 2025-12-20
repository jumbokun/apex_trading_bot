"""
Apex Exchange API Client
========================
封装 Apex Exchange API v3 的 HTTP 请求，包括签名生成和请求处理。
"""
import hmac
import base64
import hashlib
import time
import json
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from urllib.parse import urlencode
import httpx
from loguru import logger


class ApexAPIError(Exception):
    """Apex Exchange API 错误"""
    
    def __init__(self, code: str, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")


class ApexClient:
    """Apex Exchange API 客户端"""
    
    # API 端点
    REST_BASE_MAINNET = "https://omni.apex.exchange/api"
    REST_BASE_TESTNET = "https://testnet.omni.apex.exchange/api"
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: Optional[str] = None,
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
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = self.REST_BASE_TESTNET if testnet else self.REST_BASE_MAINNET
        
        self._client: Optional[httpx.Client] = None
    
    def __enter__(self):
        self._client = httpx.Client(timeout=30.0)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()
    
    def _get_timestamp(self) -> str:
        """获取 Unix 时间戳（秒）"""
        return str(int(time.time()))
    
    def _sign(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """
        生成 API 签名
        签名 = Base64(HMAC-SHA256(timestamp + method + requestPath + body, secretKey))
        
        Args:
            timestamp: Unix 时间戳
            method: HTTP 方法 (GET/POST)
            request_path: 请求路径（包含查询参数）
            body: 请求体（POST 请求）
        
        Returns:
            Base64 编码的签名
        """
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode("utf-8")
    
    def _get_headers(
        self,
        timestamp: str,
        sign: str,
        content_type: str = "application/x-www-form-urlencoded"
    ) -> Dict[str, str]:
        """构建请求头"""
        headers = {
            "APEX-API-KEY": self.api_key,
            "APEX-API-SIGNATURE": sign,
            "APEX-API-TIMESTAMP": timestamp,
            "APEX-API-PASSPHRASE": self.passphrase,
            "Content-Type": content_type,
        }
        return headers
    
    def _is_public_endpoint(self, endpoint: str) -> bool:
        """判断是否为公共 API 端点（不需要认证）"""
        return endpoint.startswith("/v3/public/")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict:
        """
        发送 API 请求
        
        Args:
            method: HTTP 方法 (GET/POST)
            endpoint: API 端点（如 /v3/account）
            params: 查询参数（GET 请求）
            data: POST 请求体数据
        
        Returns:
            API 响应的 JSON 数据
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'with' context manager.")
        
        # 构建请求路径
        request_path = endpoint
        if params:
            query_string = urlencode({k: v for k, v in params.items() if v is not None})
            if query_string:
                request_path = f"{endpoint}?{query_string}"
        
        # POST 请求体（x-www-form-urlencoded 格式）
        body = ""
        if data and method.upper() == "POST":
            body = urlencode(data)
        
        # 判断是否需要认证
        is_public = self._is_public_endpoint(endpoint)
        
        # 生成签名和请求头
        if is_public:
            # 公共 API 不需要认证，使用 JSON 格式
            headers = {}
        else:
            # 私有 API 需要认证，使用 x-www-form-urlencoded 格式
            timestamp = self._get_timestamp()
            sign = self._sign(timestamp, method, request_path, body)
            headers = self._get_headers(timestamp, sign)
        
        # 发送请求
        url = f"{self.base_url}{request_path}"
        
        try:
            if method.upper() == "GET":
                response = self._client.get(url, headers=headers)
            else:
                response = self._client.post(url, headers=headers, content=body)
            
            response.raise_for_status()
            result = response.json()
            
            # 检查 API 错误响应
            if not result.get("success", True):
                error_msg = result.get("message", "Unknown error")
                error_code = result.get("code", "unknown")
                logger.error(f"Apex API Error: {result}")
                raise ApexAPIError(
                    code=error_code,
                    message=error_msg,
                    data=result.get("data"),
                )
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """GET 请求"""
        return self._request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """POST 请求"""
        return self._request("POST", endpoint, data=data)

