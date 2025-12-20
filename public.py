"""
公共 API（市场数据、K线、深度等）
"""
from typing import Optional, Dict
from .apex_client import ApexClient


class PublicAPI:
    """公共 API（无需认证）"""
    
    def __init__(self, client: ApexClient):
        self.client = client
    
    def get_system_time(self) -> Dict:
        """
        获取系统时间
        GET /v3/public/time
        
        Returns:
            系统时间信息
        """
        return self.client.get("/v3/public/time")
    
    def get_all_config_data(self) -> Dict:
        """
        获取所有配置数据
        GET /v3/public/config
        
        Returns:
            配置数据，包含交易对信息等
        """
        return self.client.get("/v3/public/config")
    
    def get_market_depth(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> Dict:
        """
        获取市场深度
        GET /v3/public/depth
        
        Args:
            symbol: 交易对符号
            limit: 深度数量（可选）
        
        Returns:
            市场深度数据
        """
        params = {"symbol": symbol}
        if limit:
            params["limit"] = str(limit)
        
        return self.client.get("/v3/public/depth", params)
    
    def get_newest_trading_data(
        self,
        symbol: str
    ) -> Dict:
        """
        获取最新交易数据
        GET /v3/public/trade
        
        Args:
            symbol: 交易对符号
        
        Returns:
            最新交易数据
        """
        params = {"symbol": symbol}
        return self.client.get("/v3/public/trade", params)
    
    def get_candlestick_data(
        self,
        symbol: str,
        interval: str,  # "1m", "5m", "15m", "30m", "1h", "4h", "1d", etc.
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        获取 K 线数据
        GET /v3/public/kline
        
        Args:
            symbol: 交易对符号
            interval: K 线间隔（如 "1m", "5m", "15m", "30m", "1h", "4h", "1d"）
            start_time: 开始时间（Unix 时间戳，可选）
            end_time: 结束时间（Unix 时间戳，可选）
            limit: 返回数量（可选）
        
        Returns:
            K 线数据列表
        """
        params = {
            "symbol": symbol,
            "interval": interval,
        }
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if limit:
            params["limit"] = str(limit)
        
        return self.client.get("/v3/public/kline", params)
    
    def get_ticker_data(
        self,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        获取 Ticker 数据
        GET /v3/public/ticker
        
        Args:
            symbol: 交易对符号（可选，不传则返回所有交易对）
        
        Returns:
            Ticker 数据
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        return self.client.get("/v3/public/ticker", params)
    
    def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        获取资金费率历史
        GET /v3/public/funding-rate-history
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间（Unix 时间戳，可选）
            end_time: 结束时间（Unix 时间戳，可选）
            page: 页码（可选）
            limit: 每页数量（可选）
        
        Returns:
            资金费率历史数据
        """
        params = {"symbol": symbol}
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        
        return self.client.get("/v3/public/funding-rate-history", params)

