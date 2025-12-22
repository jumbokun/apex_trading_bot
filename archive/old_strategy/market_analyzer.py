"""
市场分析模块
负责扫描市场、分析量价行为、生成交易信号
"""
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
import numpy as np


@dataclass
class MarketSignal:
    """市场信号"""

    symbol: str
    timestamp: int
    signal_type: str  # "BUY", "SELL", "HOLD", "CLOSE"
    strength: float  # 信号强度 0-1
    price: float
    reason: str  # 信号原因描述
    indicators: Dict  # 技术指标值

    def __str__(self):
        return f"{self.signal_type} {self.symbol} @ {self.price} (强度: {self.strength:.2f}) - {self.reason}"


class TechnicalIndicators:
    """技术指标计算"""

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """计算简单移动平均线"""
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """计算指数移动平均线"""
        if len(prices) < period:
            return None
        prices_array = np.array(prices)
        ema = prices_array[0]
        multiplier = 2 / (period + 1)
        for price in prices_array[1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """计算相对强弱指标 RSI"""
        if len(prices) < period + 1:
            return None

        prices_array = np.array(prices)
        deltas = np.diff(prices_array)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> Optional[float]:
        """计算波动率（标准差）"""
        if len(prices) < period:
            return None
        returns = np.diff(prices[-period:]) / prices[-period:-1]
        return float(np.std(returns))

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """计算平均真实波动范围 ATR"""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None

        tr_list = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return None

        atr = np.mean(tr_list[-period:])
        return float(atr)


class MarketAnalyzer:
    """市场分析器"""

    def __init__(self, apex_exchange, strategy_config, risk_config):
        """
        初始化市场分析器

        Args:
            apex_exchange: Apex交易所实例
            strategy_config: 策略配置
            risk_config: 风险配置
        """
        self.apex = apex_exchange
        self.strategy = strategy_config
        self.risk = risk_config
        self.indicators_calc = TechnicalIndicators()

    def scan_market(self) -> List[MarketSignal]:
        """
        扫描市场，生成交易信号

        Returns:
            市场信号列表
        """
        signals = []

        for symbol in self.strategy.symbols:
            try:
                signal = self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
                    logger.info(f"生成信号: {signal}")
            except Exception as e:
                logger.error(f"分析 {symbol} 时出错: {e}")

        return signals

    def _analyze_symbol(self, symbol: str) -> Optional[MarketSignal]:
        """
        分析单个交易对

        Args:
            symbol: 交易对符号

        Returns:
            市场信号或None
        """
        # 获取K线数据
        klines_data = self.apex.public.get_candlestick_data(
            symbol=symbol,
            interval=self.strategy.timeframe,
            limit=self.strategy.kline_lookback
        )

        if not klines_data.get("success") or not klines_data.get("data"):
            logger.warning(f"{symbol} K线数据获取失败")
            return None

        klines = klines_data["data"]
        if len(klines) < self.strategy.kline_lookback:
            logger.warning(f"{symbol} K线数据不足")
            return None

        # 解析K线数据
        closes = [float(k["close"]) for k in klines]
        highs = [float(k["high"]) for k in klines]
        lows = [float(k["low"]) for k in klines]
        volumes = [float(k["volume"]) for k in klines]
        current_price = closes[-1]

        # 计算技术指标
        ma_short = self.indicators_calc.calculate_sma(closes, self.strategy.ma_short_period)
        ma_long = self.indicators_calc.calculate_sma(closes, self.strategy.ma_long_period)
        rsi = self.indicators_calc.calculate_rsi(closes, self.strategy.rsi_period)
        volatility = self.indicators_calc.calculate_volatility(closes)
        atr = self.indicators_calc.calculate_atr(highs, lows, closes)

        # 成交量分析
        volume_ma = np.mean(volumes[-self.strategy.volume_ma_period:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0

        # 检查市场是否适合交易
        if not self._is_market_tradable(symbol, volatility, volume_ratio):
            logger.info(f"{symbol} 市场条件不适合交易")
            return None

        # 生成交易信号
        indicators = {
            "ma_short": ma_short,
            "ma_long": ma_long,
            "rsi": rsi,
            "volatility": volatility,
            "atr": atr,
            "volume_ratio": volume_ratio,
        }

        signal = self._generate_signal(
            symbol=symbol,
            current_price=current_price,
            indicators=indicators
        )

        return signal

    def _is_market_tradable(self, symbol: str, volatility: float, volume_ratio: float) -> bool:
        """
        检查市场是否适合交易

        Args:
            symbol: 交易对
            volatility: 波动率
            volume_ratio: 成交量比率

        Returns:
            是否可交易
        """
        # 检查波动性
        if volatility and volatility > self.risk.max_volatility:
            logger.warning(f"{symbol} 波动性过高: {volatility:.4f}")
            return False

        # 检查成交量
        if volume_ratio < self.risk.min_volume_ratio:
            logger.warning(f"{symbol} 成交量过低: {volume_ratio:.2f}")
            return False

        # 检查资金费率
        try:
            funding_data = self.apex.account.get_funding_rate(symbol=symbol)
            if funding_data.get("success") and funding_data.get("data"):
                funding_rate = abs(float(funding_data["data"].get("fundingRate", 0)))
                if funding_rate > self.risk.max_funding_rate:
                    logger.warning(f"{symbol} 资金费率过高: {funding_rate:.6f}")
                    return False
        except Exception as e:
            logger.warning(f"获取 {symbol} 资金费率失败: {e}")

        return True

    def _generate_signal(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict
    ) -> Optional[MarketSignal]:
        """
        根据技术指标生成交易信号

        Args:
            symbol: 交易对
            current_price: 当前价格
            indicators: 技术指标字典

        Returns:
            交易信号或None
        """
        ma_short = indicators.get("ma_short")
        ma_long = indicators.get("ma_long")
        rsi = indicators.get("rsi")
        volume_ratio = indicators.get("volume_ratio", 0)

        if not all([ma_short, ma_long, rsi]):
            return None

        # 信号评分系统
        buy_score = 0
        sell_score = 0
        reasons = []

        # 趋势信号 - 移动平均线
        if ma_short > ma_long:
            buy_score += 0.3
            reasons.append(f"短期均线({ma_short:.2f})上穿长期均线({ma_long:.2f})")
        elif ma_short < ma_long:
            sell_score += 0.3
            reasons.append(f"短期均线({ma_short:.2f})下穿长期均线({ma_long:.2f})")

        # 价格位置
        if current_price > ma_short:
            buy_score += 0.1
        elif current_price < ma_short:
            sell_score += 0.1

        # RSI信号
        if rsi < self.strategy.rsi_oversold:
            buy_score += 0.3
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi > self.strategy.rsi_overbought:
            sell_score += 0.3
            reasons.append(f"RSI超买({rsi:.1f})")

        # 成交量确认
        if volume_ratio > self.strategy.volume_surge_ratio:
            if buy_score > sell_score:
                buy_score += 0.2
                reasons.append(f"成交量激增({volume_ratio:.2f}x)")
            elif sell_score > buy_score:
                sell_score += 0.2
                reasons.append(f"成交量激增({volume_ratio:.2f}x)")

        # 中性RSI区域的温和信号
        if 40 < rsi < 60:
            if ma_short > ma_long and current_price > ma_short:
                buy_score += 0.1
            elif ma_short < ma_long and current_price < ma_short:
                sell_score += 0.1

        # 确定最终信号
        signal_type = "HOLD"
        strength = 0
        reason = "无明确信号"

        if buy_score > sell_score and buy_score >= self.strategy.min_signal_strength:
            signal_type = "BUY"
            strength = buy_score
            reason = "; ".join(reasons)
        elif sell_score > buy_score and sell_score >= self.strategy.min_signal_strength:
            signal_type = "SELL"
            strength = sell_score
            reason = "; ".join(reasons)

        if signal_type == "HOLD":
            return None

        return MarketSignal(
            symbol=symbol,
            timestamp=int(time.time()),
            signal_type=signal_type,
            strength=min(strength, 1.0),
            price=current_price,
            reason=reason,
            indicators=indicators,
        )

    def check_exit_signal(self, symbol: str, position_side: str) -> Optional[MarketSignal]:
        """
        检查是否应该退出现有仓位

        Args:
            symbol: 交易对
            position_side: 仓位方向 ("LONG" or "SHORT")

        Returns:
            退出信号或None
        """
        signal = self._analyze_symbol(symbol)

        if not signal:
            return None

        # 如果持有多头且出现卖出信号
        if position_side == "LONG" and signal.signal_type == "SELL":
            signal.signal_type = "CLOSE"
            signal.reason = f"平多: {signal.reason}"
            return signal

        # 如果持有空头且出现买入信号
        if position_side == "SHORT" and signal.signal_type == "BUY":
            signal.signal_type = "CLOSE"
            signal.reason = f"平空: {signal.reason}"
            return signal

        return None
