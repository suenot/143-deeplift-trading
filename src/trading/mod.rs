//! # Trading Module
//!
//! This module contains trading signal generation and strategy implementations.

pub mod signals;
pub mod strategy;

pub use signals::{TradingSignal, SignalGenerator};
pub use strategy::DeepLiftStrategy;
