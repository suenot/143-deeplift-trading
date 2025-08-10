//! Neural network implementation for trading.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Simple feedforward neural network for trading.
#[derive(Debug, Clone)]
pub struct TradingNetwork {
    /// Weights for each layer (layer, neuron, input_weight)
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Biases for each layer (layer, neuron)
    pub biases: Vec<Vec<f64>>,
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
}

impl TradingNetwork {
    /// Create a new network with random weights.
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            let layer_weights: Vec<Vec<f64>> = (0..output_size)
                .map(|_| (0..input_size).map(|_| normal.sample(&mut rng)).collect())
                .collect();

            let layer_biases: Vec<f64> = (0..output_size).map(|_| 0.0).collect();

            weights.push(layer_weights);
            biases.push(layer_biases);
        }

        Self {
            weights,
            biases,
            layer_sizes,
        }
    }

    /// Create a network with specified weights.
    pub fn with_weights(
        layer_sizes: Vec<usize>,
        weights: Vec<Vec<Vec<f64>>>,
        biases: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            weights,
            biases,
            layer_sizes,
        }
    }

    /// Forward pass through the network.
    pub fn forward(&self, input: &[f64]) -> f64 {
        let mut current = input.to_vec();

        for (layer_idx, (layer_weights, layer_biases)) in
            self.weights.iter().zip(self.biases.iter()).enumerate()
        {
            let mut next = vec![0.0; layer_weights.len()];

            for (neuron_idx, (weights, bias)) in
                layer_weights.iter().zip(layer_biases.iter()).enumerate()
            {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(current.iter()) {
                    sum += w * x;
                }

                // Apply ReLU for hidden layers
                if layer_idx < self.weights.len() - 1 {
                    next[neuron_idx] = sum.max(0.0);
                } else {
                    next[neuron_idx] = sum;
                }
            }

            current = next;
        }

        current.get(0).copied().unwrap_or(0.0)
    }

    /// Batch forward pass.
    pub fn forward_batch(&self, inputs: &[Vec<f64>]) -> Vec<f64> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    /// Get activations for all layers.
    pub fn forward_with_activations(&self, input: &[f64]) -> (f64, Vec<Vec<f64>>) {
        let mut activations = vec![input.to_vec()];
        let mut current = input.to_vec();

        for (layer_idx, (layer_weights, layer_biases)) in
            self.weights.iter().zip(self.biases.iter()).enumerate()
        {
            let mut next = vec![0.0; layer_weights.len()];

            for (neuron_idx, (weights, bias)) in
                layer_weights.iter().zip(layer_biases.iter()).enumerate()
            {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(current.iter()) {
                    sum += w * x;
                }

                if layer_idx < self.weights.len() - 1 {
                    next[neuron_idx] = sum.max(0.0);
                } else {
                    next[neuron_idx] = sum;
                }
            }

            activations.push(next.clone());
            current = next;
        }

        let output = current.get(0).copied().unwrap_or(0.0);
        (output, activations)
    }

    /// Simple gradient descent update.
    pub fn update(&mut self, gradients: &[Vec<Vec<f64>>], bias_gradients: &[Vec<f64>], lr: f64) {
        for (layer_idx, (layer_weights, layer_grads)) in
            self.weights.iter_mut().zip(gradients.iter()).enumerate()
        {
            for (neuron_weights, neuron_grads) in layer_weights.iter_mut().zip(layer_grads.iter()) {
                for (w, g) in neuron_weights.iter_mut().zip(neuron_grads.iter()) {
                    *w -= lr * g;
                }
            }

            if let Some(layer_bias_grads) = bias_gradients.get(layer_idx) {
                for (b, g) in self.biases[layer_idx].iter_mut().zip(layer_bias_grads.iter()) {
                    *b -= lr * g;
                }
            }
        }
    }

    /// Input size.
    pub fn input_size(&self) -> usize {
        self.layer_sizes.first().copied().unwrap_or(0)
    }

    /// Output size.
    pub fn output_size(&self) -> usize {
        self.layer_sizes.last().copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = TradingNetwork::new(vec![10, 32, 16, 1]);
        assert_eq!(network.input_size(), 10);
        assert_eq!(network.output_size(), 1);
        assert_eq!(network.weights.len(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let network = TradingNetwork::new(vec![3, 4, 1]);
        let input = vec![0.1, 0.2, 0.3];
        let output = network.forward(&input);
        assert!(output.is_finite());
    }

    #[test]
    fn test_forward_with_activations() {
        let network = TradingNetwork::new(vec![3, 4, 2, 1]);
        let input = vec![0.1, 0.2, 0.3];
        let (output, activations) = network.forward_with_activations(&input);
        assert!(output.is_finite());
        assert_eq!(activations.len(), 4); // input + 3 layers
    }
}
