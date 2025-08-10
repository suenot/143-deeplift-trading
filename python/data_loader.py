"""
Data loading and feature engineering for DeepLift Trading.

This module provides:
- StockDataLoader: Fetch stock market data using yfinance
- BybitDataLoader: Fetch cryptocurrency data from Bybit API
- Feature engineering: Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Train/test data splitting utilities
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """Standard OHLCV (Open, High, Low, Close, Volume) data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def fetch_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch market data for a symbol."""
        pass

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features from raw data."""
        pass


class StockDataLoader(BaseDataLoader):
    """
    Data loader for stock market data using yfinance.

    Fetches historical price data and computes technical indicators
    for use with DeepLift Trading models.

    Example:
        >>> loader = StockDataLoader()
        >>> df = loader.fetch_data('AAPL', period='1y', interval='1d')
        >>> features, feature_names = loader.prepare_features(df)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize StockDataLoader.

        Args:
            cache_dir: Directory for caching downloaded data (optional)
        """
        self.cache_dir = cache_dir
        self._yf = None

    def _get_yfinance(self):
        """Lazy import of yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is required for StockDataLoader. "
                    "Install with: pip install yfinance"
                )
        return self._yf

    def fetch_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Start date (YYYY-MM-DD), overrides period if specified
            end: End date (YYYY-MM-DD), overrides period if specified

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        yf = self._get_yfinance()

        logger.info(f"Fetching data for {symbol}...")

        ticker = yf.Ticker(symbol)

        if start and end:
            df = ticker.history(start=start, end=end, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        # Standardize column names
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            'adj close': 'adj_close',
            'stock splits': 'stock_splits'
        })

        # Keep only OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        logger.info(f"Fetched {len(df)} rows for {symbol}")

        return df[required_cols]

    def fetch_multiple(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            **kwargs: Arguments passed to fetch_data

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_data(symbol, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        return results

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_indicators: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare technical indicator features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            include_indicators: List of indicators to include (None = all)

        Returns:
            Tuple of (features array, feature names list)
        """
        features_df = pd.DataFrame(index=df.index)

        # Price returns
        features_df['return_1d'] = df['close'].pct_change(1)
        features_df['return_5d'] = df['close'].pct_change(5)
        features_df['return_10d'] = df['close'].pct_change(10)
        features_df['return_20d'] = df['close'].pct_change(20)

        # Moving average ratios
        features_df['sma_10_ratio'] = df['close'] / df['close'].rolling(10).mean() - 1
        features_df['sma_20_ratio'] = df['close'] / df['close'].rolling(20).mean() - 1
        features_df['sma_50_ratio'] = df['close'] / df['close'].rolling(50).mean() - 1

        # EMA ratios
        features_df['ema_10_ratio'] = df['close'] / df['close'].ewm(span=10).mean() - 1
        features_df['ema_20_ratio'] = df['close'] / df['close'].ewm(span=20).mean() - 1

        # Volatility
        features_df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
        features_df['volatility_20d'] = df['close'].pct_change().rolling(20).std()

        # RSI
        features_df['rsi_14'] = self._compute_rsi(df['close'], period=14)

        # MACD
        macd, signal, hist = self._compute_macd(df['close'])
        features_df['macd'] = macd / df['close']  # Normalized
        features_df['macd_signal'] = signal / df['close']
        features_df['macd_hist'] = hist / df['close']

        # Bollinger Bands position
        features_df['bb_position'] = self._compute_bollinger_position(df['close'])

        # Volume features
        features_df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean() - 1
        features_df['volume_change'] = df['volume'].pct_change()

        # High-Low range
        features_df['hl_range'] = (df['high'] - df['low']) / df['close']

        # Momentum
        features_df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1

        # Filter indicators if specified
        if include_indicators:
            available = [col for col in include_indicators if col in features_df.columns]
            features_df = features_df[available]

        # Drop NaN rows
        features_df = features_df.dropna()

        feature_names = features_df.columns.tolist()
        features = features_df.values

        return features, feature_names

    def get_train_test_split(
        self,
        df: pd.DataFrame,
        target_horizon: int = 5,
        train_ratio: float = 0.8,
        buy_threshold: float = 0.005,
        sell_threshold: float = -0.005
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare train/test split with features and labels.

        Args:
            df: DataFrame with OHLCV data
            target_horizon: Number of periods ahead for target
            train_ratio: Fraction of data for training
            buy_threshold: Return threshold for BUY signal
            sell_threshold: Return threshold for SELL signal

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names)
        """
        # Prepare features
        features, feature_names = self.prepare_features(df)

        # Align with original data (features have NaN dropped)
        aligned_df = df.iloc[-len(features):]

        # Create future returns target
        future_returns = aligned_df['close'].pct_change(target_horizon).shift(-target_horizon)

        # Create labels (0=Sell, 1=Hold, 2=Buy)
        labels = np.ones(len(future_returns), dtype=np.int64)  # Default HOLD
        labels[future_returns > buy_threshold] = 2  # BUY
        labels[future_returns < sell_threshold] = 0  # SELL

        # Remove rows where target is NaN
        valid_mask = ~future_returns.isna().values
        features = features[valid_mask]
        labels = labels[valid_mask]

        # Split data
        split_idx = int(len(features) * train_ratio)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        return X_train, y_train, X_test, y_test, feature_names

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to [0, 1]

    @staticmethod
    def _compute_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD indicator."""
        fast_ema = prices.ewm(span=fast_period).mean()
        slow_ema = prices.ewm(span=slow_period).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def _compute_bollinger_position(
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """Compute position within Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        position = (prices - lower) / (upper - lower + 1e-10)
        return position - 0.5  # Center around 0


class BybitDataLoader(BaseDataLoader):
    """
    Data loader for cryptocurrency data from Bybit API.

    Fetches historical kline (candlestick) data from Bybit exchange.

    Example:
        >>> loader = BybitDataLoader()
        >>> df = loader.fetch_data('BTCUSDT', interval='60', limit=500)
        >>> features, feature_names = loader.prepare_features(df)
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, testnet: bool = False):
        """
        Initialize BybitDataLoader.

        Args:
            testnet: Whether to use testnet API
        """
        self.base_url = "https://api-testnet.bybit.com" if testnet else self.BASE_URL
        self.session = requests.Session()

    def fetch_data(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200,
        category: str = "spot",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical kline data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
            limit: Number of klines to fetch (max 1000)
            category: Market category ('spot', 'linear', 'inverse')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/v5/market/kline"

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"API error: {data.get('retMsg')}")

            klines = data["result"]["list"]

            # Parse klines
            records = []
            for item in klines:
                records.append({
                    'timestamp': pd.to_datetime(int(item[0]), unit='ms'),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5]),
                    'turnover': float(item[6])
                })

            df = pd.DataFrame(records)

            # Bybit returns descending order, reverse to ascending
            df = df.iloc[::-1].reset_index(drop=True)
            df.set_index('timestamp', inplace=True)

            logger.info(f"Fetched {len(df)} klines for {symbol}")

            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch klines: {e}")
            raise

    def fetch_multiple(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch klines for multiple symbols.

        Args:
            symbols: List of trading pairs
            **kwargs: Arguments passed to fetch_data

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_data(symbol, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        return results

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_indicators: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare technical indicator features from kline data.

        Uses same indicators as StockDataLoader for consistency.

        Args:
            df: DataFrame with OHLCV data
            include_indicators: List of indicators to include (None = all)

        Returns:
            Tuple of (features array, feature names list)
        """
        # Use StockDataLoader's prepare_features for consistency
        stock_loader = StockDataLoader()
        return stock_loader.prepare_features(df, include_indicators)

    def get_train_test_split(
        self,
        df: pd.DataFrame,
        target_horizon: int = 5,
        train_ratio: float = 0.8,
        buy_threshold: float = 0.005,
        sell_threshold: float = -0.005
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare train/test split with features and labels.

        Args:
            df: DataFrame with kline data
            target_horizon: Number of periods ahead for target
            train_ratio: Fraction of data for training
            buy_threshold: Return threshold for BUY signal
            sell_threshold: Return threshold for SELL signal

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names)
        """
        stock_loader = StockDataLoader()
        return stock_loader.get_train_test_split(
            df, target_horizon, train_ratio, buy_threshold, sell_threshold
        )

    def get_ticker_info(self, symbol: str, category: str = "spot") -> Dict:
        """
        Get current ticker information.

        Args:
            symbol: Trading pair
            category: Market category

        Returns:
            Dictionary with ticker info
        """
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": category, "symbol": symbol}

        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data.get('retMsg')}")

        return data["result"]["list"][0] if data["result"]["list"] else {}


class SimulatedDataGenerator:
    """
    Generate simulated market data for testing and development.

    Useful for testing strategies without needing real market data.
    """

    @staticmethod
    def generate_random_walk(
        n_periods: int,
        base_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate random walk price data.

        Args:
            n_periods: Number of periods to generate
            base_price: Starting price
            volatility: Daily volatility (standard deviation)
            drift: Daily drift (mean return)

        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(None)  # Random seed for each call

        returns = np.random.normal(drift, volatility, n_periods)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        records = []
        start_date = datetime.now() - timedelta(days=n_periods)

        for i in range(n_periods):
            price = prices[i]
            intraday_vol = volatility * 0.5

            open_price = price * (1 + np.random.normal(0, intraday_vol))
            close_price = price * (1 + np.random.normal(0, intraday_vol))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_vol)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_vol)))
            volume = np.random.exponential(1000000)

            records.append({
                'timestamp': start_date + timedelta(days=i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)

        return df

    @staticmethod
    def generate_trending_data(
        n_periods: int,
        base_price: float = 100.0,
        trend: str = 'bullish'
    ) -> pd.DataFrame:
        """
        Generate data with a clear trend.

        Args:
            n_periods: Number of periods
            base_price: Starting price
            trend: 'bullish', 'bearish', or 'sideways'

        Returns:
            DataFrame with OHLCV data
        """
        if trend == 'bullish':
            drift = 0.001
            volatility = 0.015
        elif trend == 'bearish':
            drift = -0.001
            volatility = 0.02
        else:  # sideways
            drift = 0.0
            volatility = 0.01

        return SimulatedDataGenerator.generate_random_walk(
            n_periods, base_price, volatility, drift
        )

    @staticmethod
    def generate_regime_changes(
        n_periods: int,
        base_price: float = 100.0,
        regimes: Optional[List[Tuple[int, str]]] = None
    ) -> pd.DataFrame:
        """
        Generate data with changing market regimes.

        Args:
            n_periods: Total number of periods
            base_price: Starting price
            regimes: List of (duration, regime_type) tuples

        Returns:
            DataFrame with OHLCV data
        """
        if regimes is None:
            regimes = [
                (int(n_periods * 0.3), 'bullish'),
                (int(n_periods * 0.2), 'bearish'),
                (int(n_periods * 0.25), 'sideways'),
                (int(n_periods * 0.25), 'bullish')
            ]

        all_dfs = []
        current_price = base_price

        for duration, regime in regimes:
            df = SimulatedDataGenerator.generate_trending_data(
                duration, current_price, regime
            )
            all_dfs.append(df)
            current_price = df['close'].iloc[-1]

        combined = pd.concat(all_dfs)
        combined = combined.iloc[:n_periods]  # Ensure exact length

        # Reset timestamps
        start_date = datetime.now() - timedelta(days=len(combined))
        combined.index = pd.date_range(start=start_date, periods=len(combined), freq='D')

        return combined


def create_features_from_prices(
    prices: np.ndarray,
    window: int = 20
) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature array from price series.

    Convenience function for creating features without a full DataFrame.

    Args:
        prices: Array of close prices
        window: Lookback window for indicators

    Returns:
        Tuple of (features array, feature names list)
    """
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.ones_like(prices) * 1000000
    })

    loader = StockDataLoader()
    return loader.prepare_features(df)


def normalize_features(
    features: np.ndarray,
    method: str = 'zscore',
    fit_data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize feature values.

    Args:
        features: Feature array to normalize
        method: 'zscore', 'minmax', or 'robust'
        fit_data: Data to fit normalizer on (None = use features)

    Returns:
        Tuple of (normalized features, normalization parameters)
    """
    if fit_data is None:
        fit_data = features

    params = {}

    if method == 'zscore':
        params['mean'] = np.mean(fit_data, axis=0)
        params['std'] = np.std(fit_data, axis=0) + 1e-10
        normalized = (features - params['mean']) / params['std']

    elif method == 'minmax':
        params['min'] = np.min(fit_data, axis=0)
        params['max'] = np.max(fit_data, axis=0)
        range_val = params['max'] - params['min'] + 1e-10
        normalized = (features - params['min']) / range_val

    elif method == 'robust':
        params['median'] = np.median(fit_data, axis=0)
        params['iqr'] = np.percentile(fit_data, 75, axis=0) - np.percentile(fit_data, 25, axis=0) + 1e-10
        normalized = (features - params['median']) / params['iqr']

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    params['method'] = method
    return normalized, params


def apply_normalization(features: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply pre-computed normalization parameters.

    Args:
        features: Features to normalize
        params: Normalization parameters from normalize_features

    Returns:
        Normalized features
    """
    method = params.get('method', 'zscore')

    if method == 'zscore':
        return (features - params['mean']) / params['std']
    elif method == 'minmax':
        return (features - params['min']) / (params['max'] - params['min'] + 1e-10)
    elif method == 'robust':
        return (features - params['median']) / params['iqr']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


if __name__ == "__main__":
    print("Data Loader Demo")
    print("=" * 60)

    # Demo 1: Simulated data
    print("\n1. Simulated Data Generation")
    print("-" * 40)

    sim_data = SimulatedDataGenerator.generate_regime_changes(500)
    print(f"Generated {len(sim_data)} periods of simulated data")
    print(f"Price range: ${sim_data['close'].min():.2f} - ${sim_data['close'].max():.2f}")

    # Prepare features from simulated data
    stock_loader = StockDataLoader()
    features, feature_names = stock_loader.prepare_features(sim_data)
    print(f"\nComputed {len(feature_names)} features:")
    for name in feature_names[:10]:
        print(f"  - {name}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names) - 10} more")

    print(f"\nFeature shape: {features.shape}")

    # Demo 2: Train/test split
    print("\n2. Train/Test Split")
    print("-" * 40)

    X_train, y_train, X_test, y_test, names = stock_loader.get_train_test_split(
        sim_data,
        target_horizon=5,
        train_ratio=0.8
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution (train):")
    print(f"  SELL (0): {(y_train == 0).sum()}")
    print(f"  HOLD (1): {(y_train == 1).sum()}")
    print(f"  BUY (2):  {(y_train == 2).sum()}")

    # Demo 3: Feature normalization
    print("\n3. Feature Normalization")
    print("-" * 40)

    X_train_norm, norm_params = normalize_features(X_train, method='zscore')
    X_test_norm = apply_normalization(X_test, norm_params)

    print(f"Before normalization - mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"After normalization - mean: {X_train_norm.mean():.4f}, std: {X_train_norm.std():.4f}")

    # Demo 4: Try to fetch real data (may fail without internet/API)
    print("\n4. Real Data Fetching (Demo)")
    print("-" * 40)

    # Try Bybit API
    try:
        bybit_loader = BybitDataLoader()
        btc_data = bybit_loader.fetch_data('BTCUSDT', interval='60', limit=100)
        print(f"Fetched {len(btc_data)} klines from Bybit")
        print(f"Latest BTC price: ${btc_data['close'].iloc[-1]:,.2f}")
    except Exception as e:
        print(f"Bybit fetch failed (expected if no internet): {e}")

    # Try yfinance
    try:
        stock_loader = StockDataLoader()
        aapl_data = stock_loader.fetch_data('AAPL', period='1mo', interval='1d')
        print(f"Fetched {len(aapl_data)} days of AAPL data")
        print(f"Latest AAPL price: ${aapl_data['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"yfinance fetch failed: {e}")

    print("\nDemo complete!")
