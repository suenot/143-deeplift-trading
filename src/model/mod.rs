//! # Model Module
//!
//! This module contains the DeepLift neural network implementation
//! and related structures for computing feature attributions.

pub mod deeplift;

pub use deeplift::{DeepLiftConfig, DeepLiftLayer, DeepLiftNetwork};
