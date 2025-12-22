"""
Fetch real K-line data from Binance API for backtesting
Binance has public API that doesn't require authentication for market data

Note: We use Binance data for backtesting since Apex API may have access restrictions.
The trading logic remains the same - we're just using reliable historical price data.
"""
import json
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import urllib.request
import ssl

# SSL context
ssl_context = ssl.create_default_context()


class BinanceDataFetcher:
    """Fetch historical data from Binance"""

    BASE_URL = "https://api.binance.com/api/v3"

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make HTTP GET request"""
        url = f"{self.BASE_URL}{endpoint}"
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
            return None

    def get_klines(
        self,
        symbol: str = "BTCUSDC",
        interval: str = "1m",
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000
    ) -> Optional[List]:
        """
        Get K-line/candlestick data from Binance

        Args:
            symbol: Trading pair (e.g., "BTCUSDC", "BTCUSDT")
            interval: Candle interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max number of candles (max 1000)

        Returns:
            List of candle data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return self._request("/klines", params)

    def get_historical_klines(
        self,
        symbol: str = "BTCUSDC",
        interval: str = "1m",
        days: int = 7
    ) -> Tuple[List[datetime], List[float], List[float], List[float], List[float], List[float]]:
        """
        Get historical K-line data for multiple days

        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days to fetch

        Returns:
            (timestamps, opens, highs, lows, closes, volumes)
        """
        print(f"Fetching {days} days of {interval} klines for {symbol} from Binance...")

        all_klines = []

        # Calculate time range
        end_time = int(time.time() * 1000)

        # Interval in milliseconds
        interval_map = {
            "1m": 60000,
            "3m": 180000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "2h": 7200000,
            "4h": 14400000,
            "6h": 21600000,
            "8h": 28800000,
            "12h": 43200000,
            "1d": 86400000,
        }

        interval_ms = interval_map.get(interval, 60000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        current_start = start_time
        batch_count = 0

        while current_start < end_time:
            batch_count += 1
            print(f"  Batch {batch_count}: fetching from {datetime.fromtimestamp(current_start/1000)}")

            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )

            if not klines:
                print("No more data available")
                break

            all_klines.extend(klines)
            print(f"    Got {len(klines)} candles, total: {len(all_klines)}")

            if len(klines) < 1000:
                break

            # Move to next batch (last candle timestamp + interval)
            last_ts = klines[-1][0]
            current_start = last_ts + interval_ms

            # Rate limit (Binance allows 1200 requests/min)
            time.sleep(0.1)

        if not all_klines:
            print("No kline data retrieved")
            return [], [], [], [], [], []

        # Remove duplicates and sort
        seen = set()
        unique_klines = []
        for k in all_klines:
            ts = k[0]
            if ts not in seen:
                seen.add(ts)
                unique_klines.append(k)

        unique_klines.sort(key=lambda x: x[0])

        # Parse data
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for k in unique_klines:
            try:
                ts = k[0]
                dt = datetime.fromtimestamp(ts / 1000)

                o = float(k[1])
                h = float(k[2])
                l = float(k[3])
                c = float(k[4])
                v = float(k[5])

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

        print(f"\nParsed {len(timestamps)} valid candles")
        if timestamps:
            print(f"Time range: {timestamps[0]} ~ {timestamps[-1]}")
            print(f"Price range: {min(closes):.2f} - {max(closes):.2f}")
            print(f"Average volume: {sum(volumes)/len(volumes):.2f}")

        return timestamps, opens, highs, lows, closes, volumes


def fetch_btc_data(days: int = 7, interval: str = "1m") -> Tuple[List, List, List, List, List, List]:
    """
    Convenience function to fetch BTC-USDC data

    Args:
        days: Number of days
        interval: Candle interval

    Returns:
        (timestamps, opens, highs, lows, closes, volumes)
    """
    fetcher = BinanceDataFetcher()
    return fetcher.get_historical_klines(
        symbol="BTCUSDC",
        interval=interval,
        days=days
    )


def test_fetch():
    """Test fetching real data"""
    print("=" * 60)
    print("Binance - Real Data Fetcher")
    print("=" * 60)

    fetcher = BinanceDataFetcher()

    # Get 3 days of 1-minute data
    timestamps, opens, highs, lows, closes, volumes = fetcher.get_historical_klines(
        symbol="BTCUSDC",
        interval="1m",
        days=3
    )

    if timestamps:
        print("\n" + "=" * 60)
        print("Data Summary")
        print("=" * 60)
        print(f"Total candles: {len(timestamps)}")
        print(f"First: {timestamps[0]}")
        print(f"Last:  {timestamps[-1]}")
        print(f"Open:  {opens[0]:.2f} -> {opens[-1]:.2f}")
        print(f"Close: {closes[0]:.2f} -> {closes[-1]:.2f}")
        print(f"High:  {max(highs):.2f}")
        print(f"Low:   {min(lows):.2f}")
        print(f"Avg Volume: {sum(volumes)/len(volumes):.2f}")

        # Show last 5 candles
        print("\nLast 5 candles:")
        for i in range(-5, 0):
            print(f"  {timestamps[i]} | O:{opens[i]:.2f} H:{highs[i]:.2f} L:{lows[i]:.2f} C:{closes[i]:.2f} V:{volumes[i]:.2f}")

    return timestamps, opens, highs, lows, closes, volumes


if __name__ == "__main__":
    test_fetch()
