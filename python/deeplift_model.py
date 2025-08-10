"""
DeepLift Trading Model Implementation.

This module provides:
- TradingNetwork: Neural network for trading predictions (3-class: Buy/Hold/Sell)
- DeepLiftTrading: Wrapper with attribution analysis using Captum library concepts
- Feature importance extraction and prediction explanation

Reference:
    Shrikumar, A., Greenside, P., & Kundaje, A. (2017).
    Learning Important Features Through Propagating Activation Differences.
    ICML. arXiv:1704.02685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass, field
from enum import IntEnum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSignal(IntEnum):
    """Trading signal enumeration."""
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class AttributionResult:
    """
    Attribution scores for a single prediction.

    Attributes:
        feature_names: Names of input features
        attributions: Attribution scores for each feature
        baseline_output: Model output for baseline input
        actual_output: Model output for actual input
        delta: Difference between actual and baseline outputs
        predicted_class: Predicted trading signal class
        class_probabilities: Probabilities for each class [Sell, Hold, Buy]
    """
    feature_names: List[str]
    attributions: np.ndarray
    baseline_output: np.ndarray
    actual_output: np.ndarray
    delta: float
    predicted_class: int
    class_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top n contributing features by absolute attribution value.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, attribution_score) tuples
        """
        indices = np.argsort(np.abs(self.attributions))[::-1][:n]
        return [(self.feature_names[i], float(self.attributions[i])) for i in indices]

    def positive_contributors(self) -> List[Tuple[str, float]]:
        """Get features with positive contributions (pushing towards BUY)."""
        return [
            (self.feature_names[i], float(self.attributions[i]))
            for i in range(len(self.attributions))
            if self.attributions[i] > 0
        ]

    def negative_contributors(self) -> List[Tuple[str, float]]:
        """Get features with negative contributions (pushing towards SELL)."""
        return [
            (self.feature_names[i], float(self.attributions[i]))
            for i in range(len(self.attributions))
            if self.attributions[i] < 0
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert attribution result to dictionary."""
        return {
            'feature_attributions': dict(zip(self.feature_names, self.attributions.tolist())),
            'baseline_output': self.baseline_output.tolist(),
            'actual_output': self.actual_output.tolist(),
            'delta': self.delta,
            'predicted_class': self.predicted_class,
            'predicted_signal': TradingSignal(self.predicted_class).name,
            'class_probabilities': {
                'SELL': float(self.class_probabilities[0]) if len(self.class_probabilities) > 0 else 0,
                'HOLD': float(self.class_probabilities[1]) if len(self.class_probabilities) > 1 else 0,
                'BUY': float(self.class_probabilities[2]) if len(self.class_probabilities) > 2 else 0,
            }
        }


class TradingNetwork(nn.Module):
    """
    Neural network for trading signal prediction.

    Supports 3-class classification: Buy (1), Hold (0), Sell (-1)
    Designed for interpretability with DeepLift attribution.

    Architecture:
        - Input normalization (BatchNorm)
        - Multiple hidden layers with ReLU activation
        - Dropout for regularization
        - Output layer with softmax for class probabilities

    Attributes:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes (default: 3 for Buy/Hold/Sell)
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        num_classes: int = 3,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize TradingNetwork.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (default: [128, 64, 32])
            num_classes: Number of output classes (3 for Buy/Hold/Sell)
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build network layers
        layers = []
        prev_size = input_size

        # Input batch normalization
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)
        else:
            self.input_bn = nn.Identity()

        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Input normalization
        x = self.input_bn(x)

        # Hidden layers
        x = self.hidden_layers(x)

        # Output layer
        logits = self.output_layer(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities using softmax.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Predicted class tensor of shape (batch_size,) with values -1, 0, 1
        """
        logits = self.forward(x)
        # Classes are indexed 0, 1, 2 but represent -1, 0, 1 (Sell, Hold, Buy)
        class_indices = torch.argmax(logits, dim=-1)
        return class_indices - 1  # Convert to -1, 0, 1

    def get_feature_weights(self) -> np.ndarray:
        """
        Get the weights of the first layer as initial feature importance estimate.

        Returns:
            Array of shape (input_size,) with aggregated first layer weights
        """
        first_linear = None
        for module in self.hidden_layers:
            if isinstance(module, nn.Linear):
                first_linear = module
                break

        if first_linear is not None:
            weights = first_linear.weight.detach().cpu().numpy()
            # Aggregate across all neurons in first hidden layer
            return np.abs(weights).mean(axis=0)
        return np.zeros(self.input_size)


class DeepLiftTrading:
    """
    DeepLift attribution wrapper for trading models.

    Implements DeepLift algorithm for explaining neural network predictions
    in trading context. Uses concepts from the Captum library.

    DeepLift works by:
    1. Comparing actual input to a reference (baseline) input
    2. Propagating the difference in outputs back to compute attributions
    3. Assigning contribution scores to each input feature

    Attributes:
        model: TradingNetwork model to explain
        baseline: Reference input for DeepLift comparison
        multiply_by_inputs: Whether to multiply gradients by input delta
    """

    def __init__(
        self,
        model: TradingNetwork,
        baseline: Optional[torch.Tensor] = None,
        multiply_by_inputs: bool = True
    ):
        """
        Initialize DeepLiftTrading explainer.

        Args:
            model: Trained TradingNetwork model
            baseline: Reference input tensor (default: zeros)
            multiply_by_inputs: Whether to multiply gradients by input delta
        """
        self.model = model
        self.baseline = baseline
        self.multiply_by_inputs = multiply_by_inputs
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    def set_baseline(self, baseline: torch.Tensor):
        """
        Set the reference baseline for DeepLift computation.

        Args:
            baseline: Reference input tensor of shape (1, input_size) or (input_size,)
        """
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0)
        self.baseline = baseline

    def set_baseline_from_data(self, data: np.ndarray, method: str = 'mean'):
        """
        Set baseline from training data distribution.

        Args:
            data: Training data array of shape (n_samples, input_size)
            method: 'mean', 'median', or 'zeros'
        """
        if method == 'mean':
            baseline_values = np.mean(data, axis=0)
        elif method == 'median':
            baseline_values = np.median(data, axis=0)
        elif method == 'zeros':
            baseline_values = np.zeros(data.shape[1])
        else:
            raise ValueError(f"Unknown method: {method}")

        self.baseline = torch.FloatTensor(baseline_values).unsqueeze(0)

    def _compute_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute gradients of output with respect to input.

        Args:
            input_tensor: Input tensor requiring gradients
            target_class: Target class index (0=Sell, 1=Hold, 2=Buy)

        Returns:
            Gradient tensor of same shape as input
        """
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # If no target class specified, use predicted class
        if target_class is None:
            target_class = torch.argmax(output, dim=-1)

        # Create one-hot target
        if isinstance(target_class, int):
            target = torch.zeros_like(output)
            target[:, target_class] = 1.0
        else:
            target = F.one_hot(target_class, num_classes=output.shape[-1]).float()

        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=target)

        return input_tensor.grad.detach()

    def get_attributions(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        n_steps: int = 50
    ) -> AttributionResult:
        """
        Compute DeepLift attributions for a single input.

        Uses integrated gradients approximation for more stable attributions.

        Args:
            input_tensor: Input tensor of shape (input_size,) or (1, input_size)
            target_class: Target class for attribution (None = predicted class)
            feature_names: Names for input features
            n_steps: Number of steps for integrated gradients approximation

        Returns:
            AttributionResult with feature contributions
        """
        # Ensure 2D tensor
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Get baseline
        if self.baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = self.baseline.expand_as(input_tensor)

        # Compute model outputs
        self.model.eval()
        with torch.no_grad():
            baseline_output = self.model(baseline)
            actual_output = self.model(input_tensor)
            probs = F.softmax(actual_output, dim=-1)

        # Determine target class
        if target_class is None:
            target_class = torch.argmax(actual_output, dim=-1).item()

        # Compute attributions using integrated gradients approximation
        # This provides more stable DeepLift-like attributions
        delta_input = input_tensor - baseline

        # Accumulate gradients along path from baseline to input
        total_gradients = torch.zeros_like(input_tensor)

        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            interpolated = baseline + alpha * delta_input
            grad = self._compute_gradients(interpolated, target_class)
            total_gradients += grad

        # Average gradients and multiply by delta
        avg_gradients = total_gradients / n_steps

        if self.multiply_by_inputs:
            attributions = avg_gradients * delta_input
        else:
            attributions = avg_gradients

        # Compute delta in output
        output_delta = (actual_output[0, target_class] - baseline_output[0, target_class]).item()

        # Create feature names if not provided
        num_features = input_tensor.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(num_features)]

        # Convert predicted class index to signal (-1, 0, 1)
        predicted_signal = target_class - 1

        return AttributionResult(
            feature_names=feature_names,
            attributions=attributions.squeeze().detach().numpy(),
            baseline_output=baseline_output.squeeze().detach().numpy(),
            actual_output=actual_output.squeeze().detach().numpy(),
            delta=output_delta,
            predicted_class=predicted_signal,
            class_probabilities=probs.squeeze().detach().numpy()
        )

    def explain_prediction(
        self,
        input_tensor: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a human-readable explanation for a prediction.

        Args:
            input_tensor: Input tensor to explain
            feature_names: Names for input features
            top_n: Number of top features to include in explanation

        Returns:
            Dictionary with explanation details
        """
        attribution = self.get_attributions(input_tensor, feature_names=feature_names)

        signal_name = TradingSignal(attribution.predicted_class).name
        top_features = attribution.top_features(top_n)

        # Build explanation
        explanation = {
            'prediction': signal_name,
            'confidence': float(attribution.class_probabilities.max()),
            'class_probabilities': {
                'SELL': float(attribution.class_probabilities[0]),
                'HOLD': float(attribution.class_probabilities[1]),
                'BUY': float(attribution.class_probabilities[2]),
            },
            'top_contributing_features': [
                {
                    'name': name,
                    'attribution': score,
                    'direction': 'bullish' if score > 0 else 'bearish'
                }
                for name, score in top_features
            ],
            'attribution_sum': float(np.sum(attribution.attributions)),
            'output_delta': attribution.delta
        }

        # Generate text explanation
        positive_features = [f for f, s in top_features if s > 0][:3]
        negative_features = [f for f, s in top_features if s < 0][:3]

        text_parts = [f"Predicted signal: {signal_name}"]

        if positive_features:
            text_parts.append(
                f"Bullish factors: {', '.join(positive_features)}"
            )
        if negative_features:
            text_parts.append(
                f"Bearish factors: {', '.join(negative_features)}"
            )

        explanation['text'] = '. '.join(text_parts) + '.'

        return explanation

    def batch_attributions(
        self,
        inputs: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        target_class: Optional[int] = None
    ) -> List[AttributionResult]:
        """
        Compute attributions for a batch of inputs.

        Args:
            inputs: Batch of inputs (batch_size, input_size)
            feature_names: Names for input features
            target_class: Target class for all samples (None = per-sample predicted)

        Returns:
            List of AttributionResult objects
        """
        results = []
        for i in range(inputs.shape[0]):
            result = self.get_attributions(
                inputs[i:i+1],
                target_class=target_class,
                feature_names=feature_names
            )
            results.append(result)
        return results

    def compute_feature_importance(
        self,
        data: np.ndarray,
        feature_names: List[str],
        n_samples: Optional[int] = None,
        aggregation: str = 'mean_abs'
    ) -> Dict[str, float]:
        """
        Compute aggregate feature importance across multiple samples.

        Args:
            data: Feature array (n_samples, n_features)
            feature_names: Names of features
            n_samples: Number of samples to use (None = all)
            aggregation: 'mean_abs', 'mean', or 'sum_abs'

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if n_samples is not None:
            indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
            data = data[indices]

        all_attributions = []

        for i in range(len(data)):
            input_tensor = torch.FloatTensor(data[i:i+1])
            result = self.get_attributions(input_tensor, feature_names=feature_names)
            all_attributions.append(result.attributions)

        attributions_array = np.array(all_attributions)

        if aggregation == 'mean_abs':
            importance = np.mean(np.abs(attributions_array), axis=0)
        elif aggregation == 'mean':
            importance = np.mean(attributions_array, axis=0)
        elif aggregation == 'sum_abs':
            importance = np.sum(np.abs(attributions_array), axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return dict(zip(feature_names, importance.tolist()))


def create_labels_from_returns(
    returns: np.ndarray,
    buy_threshold: float = 0.001,
    sell_threshold: float = -0.001
) -> np.ndarray:
    """
    Create 3-class labels from return values.

    Args:
        returns: Array of future returns
        buy_threshold: Threshold for BUY signal
        sell_threshold: Threshold for SELL signal

    Returns:
        Array of labels (0=Sell, 1=Hold, 2=Buy) for use with CrossEntropyLoss
    """
    labels = np.ones(len(returns), dtype=np.int64)  # Default to HOLD (1)
    labels[returns > buy_threshold] = 2  # BUY
    labels[returns < sell_threshold] = 0  # SELL
    return labels


def signals_to_class_indices(signals: np.ndarray) -> np.ndarray:
    """
    Convert trading signals (-1, 0, 1) to class indices (0, 1, 2).

    Args:
        signals: Array of trading signals (-1=Sell, 0=Hold, 1=Buy)

    Returns:
        Array of class indices (0=Sell, 1=Hold, 2=Buy)
    """
    return (signals + 1).astype(np.int64)


def class_indices_to_signals(indices: np.ndarray) -> np.ndarray:
    """
    Convert class indices (0, 1, 2) to trading signals (-1, 0, 1).

    Args:
        indices: Array of class indices (0=Sell, 1=Hold, 2=Buy)

    Returns:
        Array of trading signals (-1=Sell, 0=Hold, 1=Buy)
    """
    return indices.astype(np.int64) - 1


if __name__ == "__main__":
    print("DeepLift Trading Model Demo")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 11

    # Feature names (typical technical indicators)
    feature_names = [
        "return_1d", "return_5d", "return_10d", "sma_ratio", "ema_ratio",
        "volatility", "momentum", "rsi", "macd", "bb_position", "volume_ratio"
    ]

    # Generate synthetic features
    X = np.random.randn(n_samples, n_features) * 0.1

    # Generate synthetic returns with feature dependency
    future_returns = (
        0.3 * X[:, 0] +  # return_1d positive effect
        0.2 * X[:, 7] +   # rsi positive effect
        -0.15 * X[:, 5] + # volatility negative effect
        np.random.randn(n_samples) * 0.02
    )

    # Create 3-class labels
    y = create_labels_from_returns(future_returns, buy_threshold=0.005, sell_threshold=-0.005)

    print(f"Class distribution:")
    print(f"  SELL (0): {(y == 0).sum()}")
    print(f"  HOLD (1): {(y == 1).sum()}")
    print(f"  BUY (2):  {(y == 2).sum()}")

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create model
    model = TradingNetwork(
        input_size=n_features,
        hidden_sizes=[64, 32],
        num_classes=3,
        dropout_rate=0.2
    )

    print(f"\nModel architecture:")
    print(model)

    # Train model
    print("\nTraining model...")
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            accuracy = (torch.argmax(outputs, dim=1) == y_train_t).float().mean()
            print(f"  Epoch {epoch:3d}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    # Evaluate
    model.eval()
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_accuracy = (torch.argmax(test_outputs, dim=1) == y_test_t).float().mean()
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Create DeepLift explainer
    print("\n" + "=" * 60)
    print("DeepLift Attribution Analysis")
    print("=" * 60)

    explainer = DeepLiftTrading(model)
    explainer.set_baseline_from_data(X_train, method='mean')

    # Explain a single prediction
    sample_idx = 0
    sample = torch.FloatTensor(X_test[sample_idx:sample_idx+1])

    explanation = explainer.explain_prediction(sample, feature_names=feature_names)

    print(f"\nPrediction explanation for sample {sample_idx}:")
    print(f"  {explanation['text']}")
    print(f"  Confidence: {explanation['confidence']:.4f}")
    print(f"  Class probabilities: {explanation['class_probabilities']}")
    print("\n  Top contributing features:")
    for feat in explanation['top_contributing_features'][:5]:
        sign = "+" if feat['attribution'] > 0 else ""
        print(f"    {feat['name']}: {sign}{feat['attribution']:.6f} ({feat['direction']})")

    # Compute overall feature importance
    print("\n" + "=" * 60)
    print("Overall Feature Importance")
    print("=" * 60)

    importance = explainer.compute_feature_importance(
        X_test,
        feature_names,
        n_samples=100
    )

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_importance:
        bar = "#" * int(score * 100)
        print(f"  {name:<15} {score:.6f} {bar}")

    print("\nDemo complete!")
