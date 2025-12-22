"""
Fetch real K-line data from Apex Exchange API for backtesting
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import urllib.request
import ssl

# SSL context
ssl_context = ssl.create_default_context()

# Load .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()


class ApexDataFetcher:
    """Fetch historical data from Apex Exchange"""

    REST_BASE_TESTNET = "https://testnet.omni.apex.exchange/api"
    REST_BASE_MAINNET = "https://omni.apex.exchange/api"

    def __init__(self, testnet: bool = True):
        self.base_url = self.REST_BASE_TESTNET if testnet else self.REST_BASE_MAINNET
        self.testnet = testnet

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make HTTP GET request"""
        url = f"{self.base_url}{endpoint}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query:
                url = f"{url}?{query}"

        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'ApexTradingBot/1.0')

        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Request failed: {e}")
            return {"success": False, "error": str(e)}

    def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        data = self._request("/v3/public/config")
        if data.get("success"):
            symbols = data.get("data", {}).get("symbols", [])
            return [s.get("symbol") for s in symbols]
        return []

    def get_klines(
        self,
        symbol: str,
        interval: str = "1",  # 1, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        start_time: int = None,
        end_time: int = None,
        limit: int = 500
    ) -> Optional[List[dict]]:
        """
        Get K-line/candlestick data

        Args:
            symbol: Trading pair (e.g., "BTC-USDC")
            interval: Candle interval in minutes (1, 5, 15, 30, 60, etc.) or D/W/M
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max number of candles (max 500)

        Returns:
            List of candle data or None
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._request("/v3/public/klines", params)

        if data.get("success"):
            return data.get("data", [])
        else:
            print(f"Failed to get klines: {data}")
            return None

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1",
        days: int = 7
    ) -> Tuple[List[datetime], List[float], List[float], List[float], List[float], List[float]]:
        """
        Get historical K-line data for multiple days

        Returns:
            (timestamps, opens, highs, lows, closes, volumes)
        """
        print(f"Fetching {days} days of {interval}m klines for {symbol}...")

        all_klines = []

        # Calculate time range
        end_time = int(time.time() * 1000)

        # Interval in milliseconds
        interval_map = {
            "1": 60000,
            "5": 300000,
            "15": 900000,
            "30": 1800000,
            "60": 3600000,
            "120": 7200000,
            "240": 14400000,
            "360": 21600000,
            "720": 43200000,
            "D": 86400000,
            "W": 604800000,
        }

        interval_ms = interval_map.get(interval, 60000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        current_start = start_time

        while current_start < end_time:
            # Fetch batch
            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=500
            )

            if not klines:
                print("No more data available")
                break

            all_klines.extend(klines)
            print(f"  Fetched {len(klines)} candles, total: {len(all_klines)}")

            # Move to next batch
            if len(klines) < 500:
                break

            # Get the last candle's timestamp and move forward
            last_ts = klines[-1].get("t", 0)
            if isinstance(last_ts, str):
                last_ts = int(last_ts)
            current_start = last_ts + interval_ms

            # Rate limit
            time.sleep(0.2)

        if not all_klines:
            print("No kline data retrieved")
            return [], [], [], [], [], []

        # Sort by timestamp and remove duplicates
        all_klines = sorted(all_klines, key=lambda x: x.get("t", 0))
        seen = set()
        unique_klines = []
        for k in all_klines:
            ts = k.get("t", 0)
            if ts not in seen:
                seen.add(ts)
                unique_klines.append(k)

        # Parse data
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for k in unique_klines:
            try:
                # Timestamp
                ts = k.get("t", 0)
                if isinstance(ts, str):
                    ts = int(ts)
                dt = datetime.fromtimestamp(ts / 1000)

                # OHLCV
                o = float(k.get("o", 0))
                h = float(k.get("h", 0))
                l = float(k.get("l", 0))
                c = float(k.get("c", 0))
                v = float(k.get("v", 0))

                if o > 0 and h > 0 and l > 0 and c > 0:
                    timestamps.append(dt)
                    opens.append(o)
                    highs.append(h)
                    lows.append(l)
                    closes.append(c)
                    volumes.append(v)
            except Exception as e:
                print(f"Error parsing kline: {e}")
                continue

        print(f"Parsed {len(timestamps)} valid candles")
        if timestamps:
            print(f"Time range: {timestamps[0]} ~ {timestamps[-1]}")
            print(f"Price range: {min(closes):.2f} - {max(closes):.2f}")

        return timestamps, opens, highs, lows, closes, volumes

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """Get current ticker for a symbol"""
        data = self._request("/v3/public/ticker", {"symbol": symbol})
        if data.get("success"):
            tickers = data.get("data", [])
            for t in tickers:
                if t.get("symbol") == symbol:
                    return t
        return None


def test_fetch():
    """Test fetching real data"""
    testnet = os.getenv("APEX_TESTNET", "true").lower() == "true"

    print("=" * 60)
    print("Apex Exchange - Real Data Fetcher")
    print("=" * 60)
    print(f"Network: {'Testnet' if testnet else 'Mainnet'}")

    fetcher = ApexDataFetcher(testnet=testnet)

    # Get available symbols
    print("\n[1] Available Symbols:")
    symbols = fetcher.get_symbols()
    for s in symbols[:10]:
        print(f"  - {s}")
    if len(symbols) > 10:
        print(f"  ... and {len(symbols) - 10} more")

    # Get ticker
    symbol = "BTC-USDC"
    print(f"\n[2] Current Ticker ({symbol}):")
    ticker = fetcher.get_ticker(symbol)
    if ticker:
        print(f"  Last Price: {ticker.get('lastPrice')}")
        print(f"  24h High:   {ticker.get('highPrice24h')}")
        print(f"  24h Low:    {ticker.get('lowPrice24h')}")
        print(f"  24h Volume: {ticker.get('volume24h')}")

    # Get K-lines
    print(f"\n[3] Fetching 1-minute K-lines (last 3 days)...")
    timestamps, opens, highs, lows, closes, volumes = fetcher.get_historical_klines(
        symbol=symbol,
        interval="1",
        days=3
    )

    if timestamps:
        print(f"\n[4] Data Summary:")
        print(f"  Total candles: {len(timestamps)}")
        print(f"  First: {timestamps[0]}")
        print(f"  Last:  {timestamps[-1]}")
        print(f"  Open:  {opens[0]:.2f} -> {opens[-1]:.2f}")
        print(f"  Close: {closes[0]:.2f} -> {closes[-1]:.2f}")
        print(f"  High:  {max(highs):.2f}")
        print(f"  Low:   {min(lows):.2f}")
        print(f"  Avg Volume: {sum(volumes)/len(volumes):.2f}")

    return timestamps, opens, highs, lows, closes, volumes


if __name__ == "__main__":
    test_fetch()
