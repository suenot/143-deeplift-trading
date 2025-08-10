//! # DeepLift Implementation
//!
//! DeepLift is a method for decomposing the output prediction of a neural network
//! by comparing the activation of each neuron to its "reference activation" and
//! assigning contribution scores.
//!
//! This implementation uses the Rescale Rule for computing attributions.

use crate::{DeepLiftError, Result, TradingSignal};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for DeepLift network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLiftConfig {
    /// Input dimension (number of features)
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Reference input (baseline for attribution)
    pub reference_input: Option<Vec<f64>>,
    /// Learning rate for training
    pub learning_rate: f64,
    /// Activation function type
    pub activation: ActivationType,
    /// Attribution threshold for signal generation
    pub attribution_threshold: f64,
}

impl Default for DeepLiftConfig {
    fn default() -> Self {
        Self {
            input_dim: 10,
            hidden_dims: vec![64, 32, 16],
            output_dim: 1,
            reference_input: None,
            learning_rate: 0.001,
            activation: ActivationType::ReLU,
            attribution_threshold: 0.1,
        }
    }
}

impl DeepLiftConfig {
    /// Create a new configuration with specified dimensions
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims,
            output_dim,
            ..Default::default()
        }
    }

    /// Set the reference input for attribution computation
    pub fn with_reference(mut self, reference: Vec<f64>) -> Result<Self> {
        if reference.len() != self.input_dim {
            return Err(DeepLiftError::DimensionMismatch {
                expected: self.input_dim,
                got: reference.len(),
            });
        }
        self.reference_input = Some(reference);
        Ok(self)
    }

    /// Set the activation function
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Set the attribution threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.attribution_threshold = threshold;
        self
    }
}

/// Supported activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid function
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Linear (no activation)
    Linear,
}

impl ActivationType {
    /// Apply activation function to a value
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Linear => x,
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationType::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            ActivationType::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationType::Linear => 1.0,
        }
    }
}

/// A single layer in the DeepLift network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLiftLayer {
    /// Weight matrix
    pub weights: Array2<f64>,
    /// Bias vector
    pub biases: Array1<f64>,
    /// Activation function
    pub activation: ActivationType,
    /// Pre-activation values (for backpropagation)
    pre_activations: Option<Array1<f64>>,
    /// Post-activation values
    activations: Option<Array1<f64>>,
    /// Reference pre-activations
    ref_pre_activations: Option<Array1<f64>>,
    /// Reference activations
    ref_activations: Option<Array1<f64>>,
}

impl DeepLiftLayer {
    /// Create a new layer with random initialization
    pub fn new(input_dim: usize, output_dim: usize, activation: ActivationType) -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier/Glorot initialization
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        
        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        
        let biases = Array1::zeros(output_dim);
        
        Self {
            weights,
            biases,
            activation,
            pre_activations: None,
            activations: None,
            ref_pre_activations: None,
            ref_activations: None,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        if input.len() != self.weights.ncols() {
            return Err(DeepLiftError::DimensionMismatch {
                expected: self.weights.ncols(),
                got: input.len(),
            });
        }

        // Compute pre-activation: z = Wx + b
        let pre_act: Array1<f64> = self.weights.dot(input) + &self.biases;
        
        // Apply activation function
        let act = pre_act.mapv(|x| self.activation.apply(x));
        
        // Store for backpropagation
        self.pre_activations = Some(pre_act);
        self.activations = Some(act.clone());
        
        Ok(act)
    }

    /// Forward pass for reference input
    pub fn forward_reference(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        if input.len() != self.weights.ncols() {
            return Err(DeepLiftError::DimensionMismatch {
                expected: self.weights.ncols(),
                got: input.len(),
            });
        }

        let pre_act: Array1<f64> = self.weights.dot(input) + &self.biases;
        let act = pre_act.mapv(|x| self.activation.apply(x));
        
        self.ref_pre_activations = Some(pre_act);
        self.ref_activations = Some(act.clone());
        
        Ok(act)
    }

    /// Compute DeepLift multipliers using the Rescale Rule
    pub fn compute_multipliers(&self) -> Result<Array1<f64>> {
        let act = self.activations.as_ref()
            .ok_or_else(|| DeepLiftError::AttributionError("No activations stored".to_string()))?;
        let ref_act = self.ref_activations.as_ref()
            .ok_or_else(|| DeepLiftError::AttributionError("No reference activations".to_string()))?;
        let pre_act = self.pre_activations.as_ref()
            .ok_or_else(|| DeepLiftError::AttributionError("No pre-activations stored".to_string()))?;
        let ref_pre_act = self.ref_pre_activations.as_ref()
            .ok_or_else(|| DeepLiftError::AttributionError("No reference pre-activations".to_string()))?;

        let mut multipliers = Array1::zeros(act.len());
        
        for i in 0..act.len() {
            let delta_out = act[i] - ref_act[i];
            let delta_in = pre_act[i] - ref_pre_act[i];
            
            // Rescale rule: multiplier = delta_out / delta_in (with safeguard)
            multipliers[i] = if delta_in.abs() > 1e-10 {
                delta_out / delta_in
            } else {
                // Use derivative at reference when delta is too small
                self.activation.derivative(ref_pre_act[i])
            };
        }
        
        Ok(multipliers)
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.weights.ncols()
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.weights.nrows()
    }
}

/// DeepLift neural network for trading signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLiftNetwork {
    /// Network configuration
    config: DeepLiftConfig,
    /// Network layers
    layers: Vec<DeepLiftLayer>,
    /// Reference input (baseline)
    reference: Array1<f64>,
}

impl DeepLiftNetwork {
    /// Create a new DeepLift network from configuration
    pub fn new(config: DeepLiftConfig) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;
        
        // Create hidden layers
        for &hidden_dim in &config.hidden_dims {
            layers.push(DeepLiftLayer::new(prev_dim, hidden_dim, config.activation));
            prev_dim = hidden_dim;
        }
        
        // Create output layer (linear activation)
        layers.push(DeepLiftLayer::new(prev_dim, config.output_dim, ActivationType::Linear));
        
        // Set reference input (default to zeros if not specified)
        let reference = config.reference_input.clone()
            .map(|r| Array1::from_vec(r))
            .unwrap_or_else(|| Array1::zeros(config.input_dim));
        
        Self {
            config,
            layers,
            reference,
        }
    }

    /// Create network with custom layer architecture
    pub fn with_layers(config: DeepLiftConfig, layers: Vec<DeepLiftLayer>) -> Result<Self> {
        // Validate layer dimensions
        if layers.is_empty() {
            return Err(DeepLiftError::ConfigError("Network must have at least one layer".to_string()));
        }
        
        if layers[0].input_dim() != config.input_dim {
            return Err(DeepLiftError::ConfigError(
                format!("First layer input dim {} doesn't match config input dim {}", 
                    layers[0].input_dim(), config.input_dim)
            ));
        }
        
        let reference = config.reference_input.clone()
            .map(|r| Array1::from_vec(r))
            .unwrap_or_else(|| Array1::zeros(config.input_dim));
        
        Ok(Self {
            config,
            layers,
            reference,
        })
    }

    /// Set the reference input
    pub fn set_reference(&mut self, reference: Vec<f64>) -> Result<()> {
        if reference.len() != self.config.input_dim {
            return Err(DeepLiftError::DimensionMismatch {
                expected: self.config.input_dim,
                got: reference.len(),
            });
        }
        self.reference = Array1::from_vec(reference);
        Ok(())
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.config.input_dim {
            return Err(DeepLiftError::DimensionMismatch {
                expected: self.config.input_dim,
                got: input.len(),
            });
        }

        let mut current = Array1::from_vec(input.to_vec());
        
        for layer in &mut self.layers {
            current = layer.forward(&current)?;
        }
        
        Ok(current.to_vec())
    }

    /// Forward pass for reference input
    fn forward_reference(&mut self) -> Result<Vec<f64>> {
        let mut current = self.reference.clone();
        
        for layer in &mut self.layers {
            current = layer.forward_reference(&current)?;
        }
        
        Ok(current.to_vec())
    }

    /// Compute attributions for input features using DeepLift
    ///
    /// Returns a vector of attribution scores for each input feature,
    /// indicating how much each feature contributed to the output.
    pub fn compute_attributions(&mut self, input: &[f64]) -> Result<Vec<f64>> {
        // Forward pass for both input and reference
        let output = self.forward(input)?;
        let ref_output = self.forward_reference()?;
        
        // Compute output difference
        let output_diff: Vec<f64> = output.iter()
            .zip(ref_output.iter())
            .map(|(o, r)| o - r)
            .collect();
        
        // Backpropagate attributions through layers
        let mut attributions = Array1::from_vec(output_diff);
        
        // Process layers in reverse order
        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];
            
            // Get multipliers for this layer
            let multipliers = layer.compute_multipliers()?;
            
            // Element-wise multiply attributions by multipliers
            attributions = &attributions * &multipliers;
            
            // Backpropagate through weights
            if layer_idx > 0 {
                let weights_t = layer.weights.t();
                attributions = weights_t.dot(&attributions);
            } else {
                // For input layer, distribute attributions to input features
                let weights_t = layer.weights.t();
                attributions = weights_t.dot(&attributions);
            }
        }
        
        // Normalize by input differences
        let input_arr = Array1::from_vec(input.to_vec());
        let input_diff = &input_arr - &self.reference;
        
        let final_attributions: Vec<f64> = attributions.iter()
            .zip(input_diff.iter())
            .map(|(a, d)| {
                if d.abs() > 1e-10 {
                    a * d
                } else {
                    *a * 0.01 // Small contribution for unchanged features
                }
            })
            .collect();
        
        Ok(final_attributions)
    }

    /// Generate trading signal with explanations
    ///
    /// Returns a tuple of (TradingSignal, feature_attributions)
    pub fn explain(&mut self, input: &[f64]) -> Result<(TradingSignal, Vec<f64>)> {
        let output = self.forward(input)?;
        let attributions = self.compute_attributions(input)?;
        
        // Determine signal based on output
        let signal = if output[0] > self.config.attribution_threshold {
            TradingSignal::Buy
        } else if output[0] < -self.config.attribution_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        };
        
        Ok((signal, attributions))
    }

    /// Get the most important features based on attribution magnitude
    pub fn get_top_features(&mut self, input: &[f64], top_k: usize) -> Result<Vec<(usize, f64)>> {
        let attributions = self.compute_attributions(input)?;
        
        let mut indexed: Vec<(usize, f64)> = attributions.into_iter()
            .enumerate()
            .collect();
        
        // Sort by absolute value (descending)
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        
        Ok(indexed.into_iter().take(top_k).collect())
    }

    /// Get network configuration
    pub fn config(&self) -> &DeepLiftConfig {
        &self.config
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeepLiftConfig::default();
        assert_eq!(config.input_dim, 10);
        assert_eq!(config.output_dim, 1);
    }

    #[test]
    fn test_config_with_reference() {
        let config = DeepLiftConfig::default()
            .with_reference(vec![0.0; 10])
            .unwrap();
        assert!(config.reference_input.is_some());
    }

    #[test]
    fn test_config_reference_dimension_mismatch() {
        let result = DeepLiftConfig::default()
            .with_reference(vec![0.0; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_activation_relu() {
        let relu = ActivationType::ReLU;
        assert_eq!(relu.apply(1.0), 1.0);
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.derivative(1.0), 1.0);
        assert_eq!(relu.derivative(-1.0), 0.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let sigmoid = ActivationType::Sigmoid;
        assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_layer_creation() {
        let layer = DeepLiftLayer::new(10, 5, ActivationType::ReLU);
        assert_eq!(layer.input_dim(), 10);
        assert_eq!(layer.output_dim(), 5);
    }

    #[test]
    fn test_layer_forward() {
        let mut layer = DeepLiftLayer::new(3, 2, ActivationType::ReLU);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = layer.forward(&input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_network_creation() {
        let config = DeepLiftConfig::new(5, vec![10, 5], 1);
        let network = DeepLiftNetwork::new(config);
        assert_eq!(network.num_layers(), 3); // 2 hidden + 1 output
    }

    #[test]
    fn test_network_forward() {
        let config = DeepLiftConfig::new(5, vec![10, 5], 1);
        let mut network = DeepLiftNetwork::new(config);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = network.forward(&input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_compute_attributions() {
        let config = DeepLiftConfig::new(5, vec![10, 5], 1);
        let mut network = DeepLiftNetwork::new(config);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = network.compute_attributions(&input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 5);
    }

    #[test]
    fn test_explain() {
        let config = DeepLiftConfig::new(5, vec![10, 5], 1);
        let mut network = DeepLiftNetwork::new(config);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = network.explain(&input);
        assert!(result.is_ok());
        let (signal, attributions) = result.unwrap();
        assert!(matches!(signal, TradingSignal::Buy | TradingSignal::Hold | TradingSignal::Sell));
        assert_eq!(attributions.len(), 5);
    }

    #[test]
    fn test_get_top_features() {
        let config = DeepLiftConfig::new(5, vec![10, 5], 1);
        let mut network = DeepLiftNetwork::new(config);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = network.get_top_features(&input, 3);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }
}
