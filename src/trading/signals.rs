//! # Trading Signals
//!
//! Signal generation with DeepLift explanations.

use crate::{DeepLiftError, DeepLiftNetwork, Result};
use crate::data::bybit::FEATURE_NAMES;
use serde::{Deserialize, Serialize};

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// Strong buy signal
    Buy,
    /// Hold current position
    Hold,
    /// Strong sell signal
    Sell,
}

impl TradingSignal {
    /// Get signal strength as a numeric value
    pub fn strength(&self) -> i32 {
        match self {
            TradingSignal::Buy => 1,
            TradingSignal::Hold => 0,
            TradingSignal::Sell => -1,
        }
    }

    /// Check if signal is actionable (not Hold)
    pub fn is_actionable(&self) -> bool {
        !matches!(self, TradingSignal::Hold)
    }

    /// Get signal as string
    pub fn as_str(&self) -> &'static str {
        match self {
            TradingSignal::Buy => "BUY",
            TradingSignal::Hold => "HOLD",
            TradingSignal::Sell => "SELL",
        }
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Feature attribution with interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAttribution {
    /// Feature name
    pub name: String,
    /// Feature value
    pub value: f64,
    /// Attribution score
    pub attribution: f64,
    /// Relative importance (percentage)
    pub importance: f64,
    /// Direction of influence (positive = bullish, negative = bearish)
    pub direction: AttributionDirection,
}

/// Direction of feature influence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionDirection {
    /// Feature pushes towards buy
    Bullish,
    /// Feature is neutral
    Neutral,
    /// Feature pushes towards sell
    Bearish,
}

/// Signal explanation with feature attributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalExplanation {
    /// The trading signal
    pub signal: TradingSignal,
    /// Raw model output
    pub raw_output: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Feature attributions sorted by importance
    pub attributions: Vec<FeatureAttribution>,
    /// Summary explanation
    pub summary: String,
}

/// Signal generator using DeepLift network
pub struct SignalGenerator {
    /// DeepLift network for signal generation
    network: DeepLiftNetwork,
    /// Feature names for interpretation
    feature_names: Vec<String>,
    /// Buy threshold
    buy_threshold: f64,
    /// Sell threshold
    sell_threshold: f64,
    /// Minimum confidence for actionable signals
    min_confidence: f64,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(network: DeepLiftNetwork) -> Self {
        let input_dim = network.config().input_dim;
        let feature_names = if input_dim <= FEATURE_NAMES.len() {
            FEATURE_NAMES[..input_dim].iter().map(|s| s.to_string()).collect()
        } else {
            (0..input_dim).map(|i| format!("feature_{}", i)).collect()
        };

        Self {
            network,
            feature_names,
            buy_threshold: 0.1,
            sell_threshold: -0.1,
            min_confidence: 0.3,
        }
    }

    /// Set custom thresholds
    pub fn with_thresholds(mut self, buy: f64, sell: f64) -> Self {
        self.buy_threshold = buy;
        self.sell_threshold = sell;
        self
    }

    /// Set minimum confidence
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Set custom feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.network.config().input_dim {
            return Err(DeepLiftError::DimensionMismatch {
                expected: self.network.config().input_dim,
                got: names.len(),
            });
        }
        self.feature_names = names;
        Ok(self)
    }

    /// Generate a trading signal from features
    pub fn generate_signal(&mut self, features: &[f64]) -> Result<TradingSignal> {
        let output = self.network.forward(features)?;
        let raw_output = output[0];

        let signal = if raw_output > self.buy_threshold {
            TradingSignal::Buy
        } else if raw_output < self.sell_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        };

        Ok(signal)
    }

    /// Generate signal with full explanation
    pub fn get_signal_explanation(&mut self, features: &[f64]) -> Result<SignalExplanation> {
        // Get output and attributions
        let output = self.network.forward(features)?;
        let raw_output = output[0];
        let attributions = self.network.compute_attributions(features)?;

        // Determine signal
        let signal = if raw_output > self.buy_threshold {
            TradingSignal::Buy
        } else if raw_output < self.sell_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        };

        // Calculate confidence based on output magnitude
        let max_output = self.buy_threshold.abs().max(self.sell_threshold.abs()) * 3.0;
        let confidence = (raw_output.abs() / max_output).min(1.0);

        // Calculate total attribution magnitude for importance
        let total_attribution: f64 = attributions.iter().map(|a| a.abs()).sum();

        // Build feature attributions
        let mut feature_attributions: Vec<FeatureAttribution> = features
            .iter()
            .zip(attributions.iter())
            .enumerate()
            .map(|(i, (value, attribution))| {
                let importance = if total_attribution > 0.0 {
                    (attribution.abs() / total_attribution) * 100.0
                } else {
                    0.0
                };

                let direction = if *attribution > 0.01 {
                    AttributionDirection::Bullish
                } else if *attribution < -0.01 {
                    AttributionDirection::Bearish
                } else {
                    AttributionDirection::Neutral
                };

                FeatureAttribution {
                    name: self.feature_names.get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("feature_{}", i)),
                    value: *value,
                    attribution: *attribution,
                    importance,
                    direction,
                }
            })
            .collect();

        // Sort by importance
        feature_attributions.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap()
        });

        // Generate summary
        let summary = self.generate_summary(&signal, &feature_attributions, confidence);

        Ok(SignalExplanation {
            signal,
            raw_output,
            confidence,
            attributions: feature_attributions,
            summary,
        })
    }

    /// Generate human-readable summary
    fn generate_summary(
        &self,
        signal: &TradingSignal,
        attributions: &[FeatureAttribution],
        confidence: f64,
    ) -> String {
        let signal_str = match signal {
            TradingSignal::Buy => "BUY",
            TradingSignal::Hold => "HOLD",
            TradingSignal::Sell => "SELL",
        };

        let confidence_str = if confidence > 0.7 {
            "high"
        } else if confidence > 0.4 {
            "moderate"
        } else {
            "low"
        };

        // Get top 3 contributing features
        let top_features: Vec<String> = attributions
            .iter()
            .take(3)
            .map(|a| {
                let direction = match a.direction {
                    AttributionDirection::Bullish => "+",
                    AttributionDirection::Bearish => "-",
                    AttributionDirection::Neutral => "~",
                };
                format!("{} ({})", a.name, direction)
            })
            .collect();

        format!(
            "{} signal with {} confidence ({:.1}%). Key factors: {}",
            signal_str,
            confidence_str,
            confidence * 100.0,
            top_features.join(", ")
        )
    }

    /// Check if signal meets minimum confidence threshold
    pub fn is_confident(&mut self, features: &[f64]) -> Result<bool> {
        let explanation = self.get_signal_explanation(features)?;
        Ok(explanation.confidence >= self.min_confidence)
    }

    /// Get the underlying network
    pub fn network(&self) -> &DeepLiftNetwork {
        &self.network
    }

    /// Get mutable reference to network
    pub fn network_mut(&mut self) -> &mut DeepLiftNetwork {
        &mut self.network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeepLiftConfig;

    #[test]
    fn test_signal_strength() {
        assert_eq!(TradingSignal::Buy.strength(), 1);
        assert_eq!(TradingSignal::Hold.strength(), 0);
        assert_eq!(TradingSignal::Sell.strength(), -1);
    }

    #[test]
    fn test_signal_actionable() {
        assert!(TradingSignal::Buy.is_actionable());
        assert!(!TradingSignal::Hold.is_actionable());
        assert!(TradingSignal::Sell.is_actionable());
    }

    #[test]
    fn test_signal_display() {
        assert_eq!(format!("{}", TradingSignal::Buy), "BUY");
        assert_eq!(format!("{}", TradingSignal::Sell), "SELL");
    }

    #[test]
    fn test_signal_generator_creation() {
        let config = DeepLiftConfig::new(10, vec![20, 10], 1);
        let network = DeepLiftNetwork::new(config);
        let generator = SignalGenerator::new(network);
        assert_eq!(generator.feature_names.len(), 10);
    }

    #[test]
    fn test_generate_signal() {
        let config = DeepLiftConfig::new(5, vec![10], 1);
        let network = DeepLiftNetwork::new(config);
        let mut generator = SignalGenerator::new(network);
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let signal = generator.generate_signal(&features);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_get_signal_explanation() {
        let config = DeepLiftConfig::new(5, vec![10], 1);
        let network = DeepLiftNetwork::new(config);
        let mut generator = SignalGenerator::new(network);
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let explanation = generator.get_signal_explanation(&features);
        assert!(explanation.is_ok());
        let exp = explanation.unwrap();
        assert_eq!(exp.attributions.len(), 5);
        assert!(!exp.summary.is_empty());
    }

    #[test]
    fn test_custom_thresholds() {
        let config = DeepLiftConfig::new(5, vec![10], 1);
        let network = DeepLiftNetwork::new(config);
        let generator = SignalGenerator::new(network)
            .with_thresholds(0.5, -0.5);
        assert_eq!(generator.buy_threshold, 0.5);
        assert_eq!(generator.sell_threshold, -0.5);
    }
}
