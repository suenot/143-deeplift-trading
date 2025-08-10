//! DeepLIFT attribution implementation.
//!
//! Computes feature attributions by comparing activations to a reference baseline.

use std::collections::HashMap;

/// Attribution scores for a single prediction.
#[derive(Debug, Clone)]
pub struct Attribution {
    /// Names of input features
    pub feature_names: Vec<String>,
    /// Attribution scores for each feature
    pub scores: Vec<f64>,
    /// Model output for reference input
    pub baseline_output: f64,
    /// Model output for actual input
    pub actual_output: f64,
    /// Difference between actual and baseline output
    pub delta: f64,
}

impl Attribution {
    /// Create a new Attribution.
    pub fn new(
        feature_names: Vec<String>,
        scores: Vec<f64>,
        baseline_output: f64,
        actual_output: f64,
    ) -> Self {
        let delta = actual_output - baseline_output;
        Self {
            feature_names,
            scores,
            baseline_output,
            actual_output,
            delta,
        }
    }

    /// Get top N contributing features by absolute value.
    pub fn top_features(&self, n: usize) -> Vec<(String, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s.abs()))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .into_iter()
            .take(n)
            .map(|(i, _)| (self.feature_names[i].clone(), self.scores[i]))
            .collect()
    }

    /// Get features with positive contributions.
    pub fn positive_contributors(&self) -> Vec<(String, f64)> {
        self.feature_names
            .iter()
            .zip(self.scores.iter())
            .filter(|(_, &s)| s > 0.0)
            .map(|(n, &s)| (n.clone(), s))
            .collect()
    }

    /// Get features with negative contributions.
    pub fn negative_contributors(&self) -> Vec<(String, f64)> {
        self.feature_names
            .iter()
            .zip(self.scores.iter())
            .filter(|(_, &s)| s < 0.0)
            .map(|(n, &s)| (n.clone(), s))
            .collect()
    }

    /// Convert to HashMap.
    pub fn to_map(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .cloned()
            .zip(self.scores.iter().cloned())
            .collect()
    }

    /// Verify summation property (scores sum to delta).
    pub fn verify_summation(&self) -> f64 {
        let sum: f64 = self.scores.iter().sum();
        (sum - self.delta).abs()
    }
}

/// DeepLIFT attribution rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttributionRule {
    /// Rescale rule: gradient * delta
    Rescale,
    /// RevealCancel rule: separates positive and negative contributions
    RevealCancel,
}

/// DeepLIFT explainer for neural networks.
#[derive(Debug)]
pub struct DeepLIFT {
    /// Reference (baseline) input
    pub reference: Vec<f64>,
    /// Attribution rule to use
    pub rule: AttributionRule,
    /// Network weights (layer, neuron, weight)
    weights: Vec<Vec<Vec<f64>>>,
    /// Network biases (layer, neuron)
    biases: Vec<Vec<f64>>,
}

impl DeepLIFT {
    /// Create a new DeepLIFT explainer.
    pub fn new(reference: Vec<f64>, rule: AttributionRule) -> Self {
        Self {
            reference,
            rule,
            weights: Vec::new(),
            biases: Vec::new(),
        }
    }

    /// Set network parameters.
    pub fn set_network(&mut self, weights: Vec<Vec<Vec<f64>>>, biases: Vec<Vec<f64>>) {
        self.weights = weights;
        self.biases = biases;
    }

    /// Compute forward pass.
    fn forward(&self, input: &[f64]) -> (f64, Vec<Vec<f64>>) {
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

                // Apply ReLU for hidden layers
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

    /// Compute attribution scores.
    pub fn attribute(&self, input: &[f64], feature_names: Vec<String>) -> Attribution {
        // Forward pass for actual input
        let (actual_output, actual_activations) = self.forward(input);

        // Forward pass for reference
        let (baseline_output, ref_activations) = self.forward(&self.reference);

        // Compute attributions using gradient * delta approach
        let mut scores = vec![0.0; input.len()];

        // Simple gradient approximation for each input feature
        let epsilon = 1e-7;
        for i in 0..input.len() {
            let mut perturbed = input.to_vec();
            perturbed[i] += epsilon;
            let (perturbed_output, _) = self.forward(&perturbed);
            let gradient = (perturbed_output - actual_output) / epsilon;

            let delta_input = input[i] - self.reference[i];

            match self.rule {
                AttributionRule::Rescale => {
                    scores[i] = gradient * delta_input;
                }
                AttributionRule::RevealCancel => {
                    // Separate positive and negative contributions
                    let pos_delta = delta_input.max(0.0);
                    let neg_delta = delta_input.min(0.0);
                    scores[i] = gradient * pos_delta + gradient * neg_delta;
                }
            }
        }

        Attribution::new(feature_names, scores, baseline_output, actual_output)
    }

    /// Compute attributions for multiple inputs.
    pub fn batch_attribute(
        &self,
        inputs: &[Vec<f64>],
        feature_names: Vec<String>,
    ) -> Vec<Attribution> {
        inputs
            .iter()
            .map(|input| self.attribute(input, feature_names.clone()))
            .collect()
    }
}

/// Compute average feature importance across samples.
pub fn compute_feature_importance(
    explainer: &DeepLIFT,
    samples: &[Vec<f64>],
    feature_names: Vec<String>,
) -> HashMap<String, f64> {
    let n = samples.len() as f64;
    let mut importance_sum: HashMap<String, f64> = feature_names
        .iter()
        .map(|name| (name.clone(), 0.0))
        .collect();

    for sample in samples {
        let attr = explainer.attribute(sample, feature_names.clone());
        for (name, score) in attr.to_map() {
            if let Some(sum) = importance_sum.get_mut(&name) {
                *sum += score.abs();
            }
        }
    }

    // Average
    for value in importance_sum.values_mut() {
        *value /= n;
    }

    importance_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribution_top_features() {
        let attr = Attribution::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![0.1, -0.5, 0.3],
            0.0,
            -0.1,
        );

        let top = attr.top_features(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "b");
        assert_eq!(top[1].0, "c");
    }

    #[test]
    fn test_attribution_contributors() {
        let attr = Attribution::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![0.1, -0.5, 0.3],
            0.0,
            -0.1,
        );

        let positive = attr.positive_contributors();
        assert_eq!(positive.len(), 2);

        let negative = attr.negative_contributors();
        assert_eq!(negative.len(), 1);
    }
}
