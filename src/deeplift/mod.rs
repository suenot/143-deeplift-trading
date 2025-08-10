//! DeepLIFT attribution module.
//!
//! Provides neural network interpretability through activation difference propagation.

pub mod attribution;

pub use attribution::{Attribution, DeepLIFT};
