//! # DeepLift Trading
//!
//! A Rust implementation of DeepLift attribution method for explainable trading strategies.
//!
//! DeepLift is a method for decomposing the output prediction of a neural network
//! on a specific input by backpropagating the contributions of all neurons in the
//! network to every feature of the input.
//!
//! ## Features
//!
//! - **DeepLift Attribution**: Compute feature attributions using the rescale rule
//! - **Trading Signals**: Generate explainable buy/sell signals
//! - **Backtesting**: Test strategies with historical data
//! - **Bybit Integration**: Fetch market data from Bybit exchange
//!
//! ## Example
//!
//! ```rust,no_run
//! use deeplift_trading::prelude::*;
//!
//! // Create a DeepLift network
//! let config = DeepLiftConfig::default();
//! let network = DeepLiftNetwork::new(config);
//!
//! // Generate trading signals with explanations
//! let features = vec![0.5, 0.3, -0.2, 0.8, 0.1];
//! let (signal, attributions) = network.explain(&features).unwrap();
//!
//! println!("Signal: {:?}", signal);
//! println!("Feature attributions: {:?}", attributions);
//! ```

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

// Re-exports for convenience
pub use model::deeplift::{DeepLiftConfig, DeepLiftLayer, DeepLiftNetwork};
pub use data::bybit::BybitClient;
pub use trading::signals::{TradingSignal, SignalGenerator};
pub use trading::strategy::DeepLiftStrategy;
pub use backtest::engine::{BacktestEngine, BacktestResult, BacktestConfig};

use thiserror::Error;

/// Errors that can occur in the DeepLift trading system
#[derive(Error, Debug)]
pub enum DeepLiftError {
    /// Error during network forward pass
    #[error("Forward pass error: {0}")]
    ForwardError(String),

    /// Error during attribution computation
    #[error("Attribution error: {0}")]
    AttributionError(String),

    /// Invalid network configuration
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Data fetching error
    #[error("Data error: {0}")]
    DataError(String),

    /// Network request error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Invalid input dimensions
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Trading strategy error
    #[error("Strategy error: {0}")]
    StrategyError(String),

    /// Backtest error
    #[error("Backtest error: {0}")]
    BacktestError(String),

    /// Insufficient data for computation
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Result type alias for DeepLift operations
pub type Result<T> = std::result::Result<T, DeepLiftError>;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        DeepLiftConfig, DeepLiftLayer, DeepLiftNetwork,
        BybitClient,
        TradingSignal, SignalGenerator,
        DeepLiftStrategy,
        BacktestEngine, BacktestResult, BacktestConfig,
        DeepLiftError, Result,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DeepLiftError::ConfigError("invalid layer size".to_string());
        assert!(err.to_string().contains("invalid layer size"));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = DeepLiftError::DimensionMismatch { expected: 10, got: 5 };
        assert!(err.to_string().contains("expected 10"));
        assert!(err.to_string().contains("got 5"));
    }
}
