"""
Fetch K-line data for multiple trading pairs from Binance
For FOMO strategy we need to scan hot pairs, not just BTC
"""
import json
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import urllib.request
import ssl

ssl_context = ssl.create_default_context()


class MultiPairFetcher:
    """Fetch data for multiple trading pairs"""

    BASE_URL = "https://api.binance.com/api/v3"

    # Top perpetual futures pairs (similar to Apex)
    # Using USDC pairs where available, otherwise USDT
    TOP_PAIRS = [
        "BTCUSDC", "ETHUSDC", "SOLUSDC",
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT",
        "NEARUSDT", "OPUSDT", "ARBUSDT", "SUIUSDT", "SEIUSDT",
        "TIAUSDT", "INJUSDT", "WLDUSDT", "ORDIUSDT", "BONKUSDT",
    ]

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
            return None

    def get_24h_tickers(self) -> List[dict]:
        """Get 24h ticker data for all symbols"""
        data = self._request("/ticker/24hr")
        if data:
            # Filter for our pairs
            return [t for t in data if t.get('symbol') in self.TOP_PAIRS]
        return []

    def get_hot_pairs(self, top_n: int = 10) -> List[dict]:
        """
        Get hottest pairs based on:
        - Price change %
        - Volume
        - Volatility

        Returns list of pairs sorted by "hotness"
        """
        tickers = self.get_24h_tickers()
        if not tickers:
            return []

        # Calculate hotness score
        for t in tickers:
            try:
                price_change = abs(float(t.get('priceChangePercent', 0)))
                volume = float(t.get('quoteVolume', 0))

                # Volatility proxy: (high - low) / low
                high = float(t.get('highPrice', 0))
                low = float(t.get('lowPrice', 1))
                volatility = (high - low) / low * 100 if low > 0 else 0

                # Hotness = weighted combination
                t['hotness'] = (
                    price_change * 0.4 +       # Price change importance
                    min(volume / 1e8, 10) +    # Volume (capped)
                    volatility * 0.3            # Volatility importance
                )
                t['priceChangeFloat'] = price_change
                t['volatility'] = volatility
            except:
                t['hotness'] = 0

        # Sort by hotness
        tickers.sort(key=lambda x: x.get('hotness', 0), reverse=True)

        return tickers[:top_n]

    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000
    ) -> Optional[List]:
        """Get K-line data for a symbol"""
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
        symbol: str,
        interval: str = "1m",
        days: int = 3
    ) -> Tuple[List[datetime], List[float], List[float], List[float], List[float], List[float]]:
        """Get historical K-line data"""
        all_klines = []

        end_time = int(time.time() * 1000)
        interval_map = {
            "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
            "30m": 1800000, "1h": 3600000, "4h": 14400000, "1d": 86400000,
        }
        interval_ms = interval_map.get(interval, 60000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        current_start = start_time

        while current_start < end_time:
            klines = self.get_klines(symbol, interval, current_start, end_time, 1000)
            if not klines:
                break
            all_klines.extend(klines)
            if len(klines) < 1000:
                break
            current_start = klines[-1][0] + interval_ms
            time.sleep(0.05)

        if not all_klines:
            return [], [], [], [], [], []

        # Parse
        timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        seen = set()

        for k in sorted(all_klines, key=lambda x: x[0]):
            ts = k[0]
            if ts in seen:
                continue
            seen.add(ts)

            try:
                timestamps.append(datetime.fromtimestamp(ts / 1000))
                opens.append(float(k[1]))
                highs.append(float(k[2]))
                lows.append(float(k[3]))
                closes.append(float(k[4]))
                volumes.append(float(k[5]))
            except:
                continue

        return timestamps, opens, highs, lows, closes, volumes

    def fetch_all_pairs_data(
        self,
        pairs: List[str] = None,
        interval: str = "1m",
        days: int = 3
    ) -> Dict[str, Tuple]:
        """
        Fetch K-line data for multiple pairs

        Returns:
            Dict[symbol, (timestamps, opens, highs, lows, closes, volumes)]
        """
        if pairs is None:
            pairs = self.TOP_PAIRS

        print(f"Fetching {interval} data for {len(pairs)} pairs ({days} days)...")

        all_data = {}
        for i, symbol in enumerate(pairs):
            print(f"  [{i+1}/{len(pairs)}] {symbol}...", end=" ")
            try:
                data = self.get_historical_klines(symbol, interval, days)
                if data[0]:  # has timestamps
                    all_data[symbol] = data
                    print(f"OK ({len(data[0])} candles)")
                else:
                    print("No data")
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(0.1)

        print(f"\nFetched data for {len(all_data)} pairs")
        return all_data


def test_multi_pairs():
    """Test fetching multiple pairs"""
    print("=" * 70)
    print("Multi-Pair Data Fetcher")
    print("=" * 70)

    fetcher = MultiPairFetcher()

    # Get hot pairs
    print("\n[1] Scanning for hot pairs...")
    hot_pairs = fetcher.get_hot_pairs(top_n=15)

    if hot_pairs:
        print(f"\nTop 15 Hot Pairs (by 24h activity):")
        print("-" * 70)
        print(f"{'Symbol':<12} {'Price Chg':<12} {'Volatility':<12} {'Volume(M)':<12} {'Hotness':<10}")
        print("-" * 70)
        for p in hot_pairs:
            symbol = p.get('symbol', '')
            pct = p.get('priceChangeFloat', 0)
            vol = float(p.get('quoteVolume', 0)) / 1e6
            volatility = p.get('volatility', 0)
            hotness = p.get('hotness', 0)
            print(f"{symbol:<12} {pct:+.2f}%{'':4} {volatility:.2f}%{'':5} {vol:.1f}M{'':5} {hotness:.2f}")

    # Fetch data for top 5
    print("\n[2] Fetching K-line data for top 5 hot pairs...")
    top_5_symbols = [p.get('symbol') for p in hot_pairs[:5]]

    all_data = fetcher.fetch_all_pairs_data(
        pairs=top_5_symbols,
        interval="1m",
        days=3
    )

    # Summary
    print("\n[3] Data Summary:")
    print("-" * 70)
    for symbol, data in all_data.items():
        timestamps, opens, highs, lows, closes, volumes = data
        if timestamps:
            price_change = (closes[-1] - closes[0]) / closes[0] * 100
            volatility = (max(highs) - min(lows)) / min(lows) * 100
            print(f"  {symbol:<12} {len(timestamps):>6} candles | "
                  f"Price: {closes[0]:.2f} -> {closes[-1]:.2f} ({price_change:+.2f}%) | "
                  f"Range: {volatility:.2f}%")

    return all_data


if __name__ == "__main__":
    test_multi_pairs()
