"""
DeepLIFT implementation for trading model interpretation.

This module provides:
- DeepLIFT: Attribution method for explaining predictions
- Attribution: Data class for attribution results
- TradingModelWithDeepLIFT: Neural network for trading
- compute_feature_importance: Aggregate feature importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Attribution:
    """Attribution scores for a single prediction."""
    feature_names: List[str]
    scores: np.ndarray
    baseline_output: float
    actual_output: float
    delta: float

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top n contributing features by absolute value."""
        indices = np.argsort(np.abs(self.scores))[::-1][:n]
        return [(self.feature_names[i], self.scores[i]) for i in indices]

    def positive_contributors(self) -> List[Tuple[str, float]]:
        """Get features with positive contributions."""
        return [
            (self.feature_names[i], self.scores[i])
            for i in range(len(self.scores))
            if self.scores[i] > 0
        ]

    def negative_contributors(self) -> List[Tuple[str, float]]:
        """Get features with negative contributions."""
        return [
            (self.feature_names[i], self.scores[i])
            for i in range(len(self.scores))
            if self.scores[i] < 0
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return dict(zip(self.feature_names, self.scores))


class DeepLIFT:
    """
    DeepLIFT attribution for neural network trading models.

    This implementation supports both the Rescale rule and
    RevealCancel rule for ReLU-like activations.

    Reference:
        Shrikumar, A., Greenside, P., & Kundaje, A. (2017).
        Learning Important Features Through Propagating Activation Differences.
        ICML. arXiv:1704.02685
    """

    def __init__(
        self,
        model: nn.Module,
        reference: Optional[torch.Tensor] = None,
        rule: str = "rescale"
    ):
        """
        Initialize DeepLIFT explainer.

        Args:
            model: Neural network model to explain
            reference: Reference input (baseline). If None, uses zeros.
            rule: Attribution rule - "rescale" or "reveal_cancel"
        """
        self.model = model
        self.reference = reference
        self.rule = rule

        if rule not in ["rescale", "reveal_cancel"]:
            raise ValueError(f"Unknown rule: {rule}. Use 'rescale' or 'reveal_cancel'")

    def set_reference(self, reference: torch.Tensor):
        """Update the reference baseline."""
        self.reference = reference

    def attribute(
        self,
        input_tensor: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Attribution:
        """
        Compute DeepLIFT attribution scores.

        Args:
            input_tensor: Input to explain (batch_size=1 or 1D tensor)
            feature_names: Names for input features

        Returns:
            Attribution object with contribution scores
        """
        # Ensure 2D tensor
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Set reference
        if self.reference is None:
            reference = torch.zeros_like(input_tensor)
        else:
            reference = self.reference
            if reference.dim() == 1:
                reference = reference.unsqueeze(0)
            reference = reference.expand_as(input_tensor)

        # Compute outputs
        self.model.eval()
        with torch.no_grad():
            ref_output = self.model(reference)
            actual_output = self.model(input_tensor)

        # Compute attribution using gradient * delta approach
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)

        # Backward pass
        output.backward(torch.ones_like(output))
        gradients = input_tensor.grad.detach()

        # Compute delta from reference
        delta_input = input_tensor.detach() - reference

        # DeepLIFT attribution
        if self.rule == "rescale":
            attributions = gradients * delta_input
        else:
            attributions = self._reveal_cancel_attribution(
                input_tensor.detach(), reference, gradients
            )

        # Create feature names if not provided
        num_features = input_tensor.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(num_features)]

        return Attribution(
            feature_names=feature_names,
            scores=attributions.squeeze().numpy(),
            baseline_output=ref_output.squeeze().item(),
            actual_output=actual_output.squeeze().item(),
            delta=actual_output.squeeze().item() - ref_output.squeeze().item()
        )

    def _reveal_cancel_attribution(
        self,
        input_tensor: torch.Tensor,
        reference: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attribution using RevealCancel rule.
        Separates positive and negative contributions for better attribution.
        """
        delta = input_tensor - reference

        # Separate positive and negative deltas
        positive_delta = F.relu(delta)
        negative_delta = -F.relu(-delta)

        # Compute separate attributions
        positive_attr = gradients * positive_delta
        negative_attr = gradients * negative_delta

        return positive_attr + negative_attr

    def batch_attribute(
        self,
        inputs: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> List[Attribution]:
        """
        Compute attributions for a batch of inputs.

        Args:
            inputs: Batch of inputs (batch_size, num_features)
            feature_names: Names for input features

        Returns:
            List of Attribution objects
        """
        attributions = []
        for i in range(inputs.shape[0]):
            attr = self.attribute(inputs[i:i+1], feature_names)
            attributions.append(attr)
        return attributions


class TradingModelWithDeepLIFT(nn.Module):
    """
    Neural network for trading signal prediction.
    Designed for use with DeepLIFT attribution.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize trading model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of outputs (1 for regression/binary)
            dropout: Dropout probability
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


def compute_feature_importance(
    model: nn.Module,
    features: np.ndarray,
    feature_names: List[str],
    reference: Optional[np.ndarray] = None,
    n_samples: int = 1000
) -> Dict[str, float]:
    """
    Compute average feature importance across samples.

    Args:
        model: Trained trading model
        features: Feature array (N, num_features)
        feature_names: Names of features
        reference: Reference input (default: mean of features)
        n_samples: Maximum samples to use

    Returns:
        Dictionary mapping feature names to average absolute importance
    """
    if reference is None:
        ref_tensor = torch.FloatTensor(np.mean(features, axis=0, keepdims=True))
    else:
        ref_tensor = torch.FloatTensor(reference)
        if ref_tensor.dim() == 1:
            ref_tensor = ref_tensor.unsqueeze(0)

    explainer = DeepLIFT(model, reference=ref_tensor)

    # Compute attributions for samples
    importance_sum = np.zeros(len(feature_names))
    actual_samples = min(len(features), n_samples)

    for i in range(actual_samples):
        input_tensor = torch.FloatTensor(features[i:i+1])
        attr = explainer.attribute(input_tensor, feature_names)
        importance_sum += np.abs(attr.scores)

    # Average importance
    avg_importance = importance_sum / actual_samples

    return dict(zip(feature_names, avg_importance))


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    lr: float = 0.001,
    verbose: bool = True
) -> List[float]:
    """
    Train the trading model.

    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress

    Returns:
        List of training losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    if y_train_t.dim() == 1:
        y_train_t = y_train_t.unsqueeze(1)

    losses = []
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and epoch % 20 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return losses


if __name__ == "__main__":
    # Example usage
    print("DeepLIFT Trading Example")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 11

    # Simulated features
    X = np.random.randn(n_samples, n_features) * 0.1
    # Target with some feature dependencies
    y = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.15 * X[:, 5] + np.random.randn(n_samples) * 0.05

    # Feature names
    feature_names = [
        "return_1d", "return_5d", "return_10d", "sma_ratio", "ema_ratio",
        "volatility", "momentum", "rsi", "macd", "bb_position", "volume_ratio"
    ]

    # Train/test split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train model
    model = TradingModelWithDeepLIFT(input_size=n_features, hidden_size=32)
    print("\nTraining model...")
    train_model(model, X_train, y_train, epochs=100)

    # Create DeepLIFT explainer with mean reference
    reference = torch.FloatTensor(np.mean(X_train, axis=0, keepdims=True))
    explainer = DeepLIFT(model, reference=reference)

    # Explain a single prediction
    print("\n" + "=" * 50)
    print("Single Prediction Explanation")
    print("=" * 50)

    sample_idx = 0
    sample = torch.FloatTensor(X_test[sample_idx:sample_idx+1])
    attribution = explainer.attribute(sample, feature_names)

    print(f"Baseline output: {attribution.baseline_output:.6f}")
    print(f"Actual output: {attribution.actual_output:.6f}")
    print(f"Delta: {attribution.delta:.6f}")
    print(f"\nSum of attributions: {np.sum(attribution.scores):.6f}")
    print("\nTop contributing features:")
    for name, score in attribution.top_features(5):
        direction = "+" if score > 0 else ""
        print(f"  {name}: {direction}{score:.6f}")

    # Compute overall feature importance
    print("\n" + "=" * 50)
    print("Overall Feature Importance")
    print("=" * 50)

    importance = compute_feature_importance(model, X_test, feature_names, n_samples=100)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for name, score in sorted_importance:
        print(f"  {name}: {score:.6f}")
