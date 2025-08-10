"""
DeepLIFT Trading - Neural Network Interpretability for Trading

This module provides:
- DeepLIFT: Attribution method for explaining neural network predictions
- TradingModel: Neural network for trading signal prediction
- BybitClient: Data fetching from Bybit cryptocurrency exchange
- FeatureGenerator: Technical indicator computation
- DeepLIFTBacktester: Backtesting with explanation logging

Modules:
- deeplift_trader: Core DeepLIFT implementation and trading model
- data_loader: Data fetching and feature engineering
- backtest: Backtesting framework with explanations
"""

from .deeplift_trader import (
    DeepLIFT,
    Attribution,
    TradingModelWithDeepLIFT,
    compute_feature_importance,
)

from .data_loader import (
    BybitClient,
    FeatureGenerator,
    SimulatedDataGenerator,
    create_trading_features,
)

from .backtest import (
    DeepLIFTBacktester,
    calculate_metrics,
)

__version__ = "0.1.0"
__all__ = [
    "DeepLIFT",
    "Attribution",
    "TradingModelWithDeepLIFT",
    "compute_feature_importance",
    "BybitClient",
    "FeatureGenerator",
    "SimulatedDataGenerator",
    "create_trading_features",
    "DeepLIFTBacktester",
    "calculate_metrics",
]
