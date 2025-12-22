"""
交易相关 API（下单、取消订单、查询订单等）
"""
from typing import Optional, Dict
from .apex_client import ApexClient


class TradeAPI:
    """交易相关 API"""
    
    def __init__(self, client: ApexClient):
        self.client = client
    
    def create_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        type: str,  # "LIMIT", "MARKET", etc.
        size: str,
        price: Optional[str] = None,
        client_order_id: Optional[str] = None,
        time_in_force: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        post_only: Optional[bool] = None,
        **kwargs
    ) -> Dict:
        """
        创建订单
        POST /v3/order
        
        Args:
            symbol: 交易对符号（如 BTC-USDC）
            side: 订单方向 ("BUY" 或 "SELL")
            type: 订单类型 ("LIMIT", "MARKET", etc.)
            size: 订单数量
            price: 订单价格（限价单必需）
            client_order_id: 客户端订单 ID（可选）
            time_in_force: 订单有效期（可选，如 "GTC", "IOC", "FOK"）
            reduce_only: 是否只减仓（可选）
            post_only: 是否只做 Maker（可选）
            **kwargs: 其他订单参数
        
        Returns:
            订单创建结果
        """
        data = {
            "symbol": symbol,
            "side": side,
            "type": type,
            "size": size,
        }
        
        if price:
            data["price"] = price
        if client_order_id:
            data["clientOrderId"] = client_order_id
        if time_in_force:
            data["timeInForce"] = time_in_force
        if reduce_only is not None:
            data["reduceOnly"] = str(reduce_only).lower()
        if post_only is not None:
            data["postOnly"] = str(post_only).lower()
        
        data.update(kwargs)
        
        return self.client.post("/v3/order", data)
    
    def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        取消订单
        POST /v3/order/cancel
        
        Args:
            order_id: 订单 ID
            symbol: 交易对符号（可选）
        
        Returns:
            取消结果
        """
        data = {"orderId": order_id}
        if symbol:
            data["symbol"] = symbol
        
        return self.client.post("/v3/order/cancel", data)
    
    def cancel_order_by_client_id(
        self,
        client_order_id: str,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        通过客户端订单 ID 取消订单
        POST /v3/order/cancel-by-client-order-id
        
        Args:
            client_order_id: 客户端订单 ID
            symbol: 交易对符号（可选）
        
        Returns:
            取消结果
        """
        data = {"clientOrderId": client_order_id}
        if symbol:
            data["symbol"] = symbol
        
        return self.client.post("/v3/order/cancel-by-client-order-id", data)
    
    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        获取当前挂单
        GET /v3/order/open
        
        Args:
            symbol: 交易对符号（可选）
            page: 页码
            limit: 每页数量
        
        Returns:
            当前挂单列表
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        
        return self.client.get("/v3/order/open", params)
    
    def cancel_all_orders(
        self,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        取消所有挂单
        POST /v3/order/cancel-all
        
        Args:
            symbol: 交易对符号（可选，不传则取消所有交易对的订单）
        
        Returns:
            取消结果
        """
        data = {}
        if symbol:
            data["symbol"] = symbol
        
        return self.client.post("/v3/order/cancel-all", data)
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict:
        """
        获取所有订单历史
        GET /v3/order/history
        
        Args:
            symbol: 交易对符号（可选）
            page: 页码
            limit: 每页数量
            start_time: 开始时间（Unix 时间戳）
            end_time: 结束时间（Unix 时间戳）
        
        Returns:
            订单历史列表
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        
        return self.client.get("/v3/order/history", params)
    
    def get_order_by_id(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        通过订单 ID 查询订单
        GET /v3/order
        
        Args:
            order_id: 订单 ID
            symbol: 交易对符号（可选）
        
        Returns:
            订单信息
        """
        params = {"orderId": order_id}
        if symbol:
            params["symbol"] = symbol
        
        return self.client.get("/v3/order", params)
    
    def get_order_by_client_id(
        self,
        client_order_id: str,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        通过客户端订单 ID 查询订单
        GET /v3/order/by-client-order-id
        
        Args:
            client_order_id: 客户端订单 ID
            symbol: 交易对符号（可选）
        
        Returns:
            订单信息
        """
        params = {"clientOrderId": client_order_id}
        if symbol:
            params["symbol"] = symbol
        
        return self.client.get("/v3/order/by-client-order-id", params)
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict:
        """
        获取成交历史
        GET /v3/trade
        
        Args:
            symbol: 交易对符号（可选）
            page: 页码
            limit: 每页数量
            start_time: 开始时间（Unix 时间戳）
            end_time: 结束时间（Unix 时间戳）
        
        Returns:
            成交历史列表
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if page:
            params["page"] = str(page)
        if limit:
            params["limit"] = str(limit)
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        
        return self.client.get("/v3/trade", params)
    
    def get_worst_price(
        self,
        symbol: str,
        size: str,
        side: str  # "BUY" or "SELL"
    ) -> Dict:
        """
        获取最差价格
        GET /v3/order/worst-price
        
        Args:
            symbol: 交易对符号
            size: 订单数量
            side: 订单方向 ("BUY" 或 "SELL")
        
        Returns:
            最差价格信息
        """
        params = {
            "symbol": symbol,
            "size": size,
            "side": side,
        }
        
        return self.client.get("/v3/order/worst-price", params)


