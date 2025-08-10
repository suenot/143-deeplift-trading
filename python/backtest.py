"""
Backtesting framework with DeepLift attribution analysis.

This module provides:
- Backtester: Main backtesting class with run_backtest() method
- Performance metrics: Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate
- Baseline strategy comparison (Buy & Hold)
- Attribution analysis during backtesting
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a trading strategy.

    Attributes:
        total_return: Total percentage return
        annualized_return: Annualized return
        annualized_volatility: Annualized volatility
        sharpe_ratio: Risk-adjusted return (excess return / volatility)
        sortino_ratio: Downside risk-adjusted return
        max_drawdown: Maximum peak-to-trough decline
        win_rate: Percentage of profitable trades
        profit_factor: Gross profit / Gross loss
        num_trades: Total number of trades executed
        avg_trade_return: Average return per trade
        avg_win: Average winning trade return
        avg_loss: Average losing trade return
        max_consecutive_wins: Longest winning streak
        max_consecutive_losses: Longest losing streak
        calmar_ratio: Annualized return / Max drawdown
    """
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    calmar_ratio: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'num_trades': self.num_trades,
            'avg_trade_return': self.avg_trade_return,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'calmar_ratio': self.calmar_ratio,
        }


@dataclass
class BacktestResult:
    """
    Complete results from a backtest run.

    Attributes:
        results_df: DataFrame with detailed trade history
        metrics: Performance metrics
        baseline_metrics: Buy & Hold baseline metrics
        feature_importance: Average feature importance from attributions
        attribution_history: List of attributions for each decision
    """
    results_df: pd.DataFrame
    metrics: PerformanceMetrics
    baseline_metrics: Optional[PerformanceMetrics] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    attribution_history: List[Dict] = field(default_factory=list)


class Backtester:
    """
    Backtesting framework with DeepLift attribution analysis.

    Runs a trading strategy simulation with:
    - Transaction cost modeling
    - Position management (Long, Hold, Short)
    - Performance metric calculation
    - Feature attribution logging
    - Baseline strategy comparison

    Example:
        >>> backtester = Backtester(
        ...     model=trained_model,
        ...     explainer=deeplift_explainer,
        ...     feature_names=feature_names,
        ...     transaction_cost=0.001
        ... )
        >>> result = backtester.run_backtest(prices, features)
        >>> print(result.metrics.sharpe_ratio)
    """

    def __init__(
        self,
        model: nn.Module,
        explainer: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        initial_capital: float = 10000.0,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Backtester.

        Args:
            model: Trained trading model (TradingNetwork)
            explainer: DeepLiftTrading explainer (optional)
            feature_names: Names of input features
            transaction_cost: Cost per transaction as fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage as fraction of price
            initial_capital: Starting capital
            periods_per_year: Number of trading periods per year (252 for daily, 8760 for hourly)
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names or []
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        compute_attributions: bool = True,
        compute_baseline: bool = True
    ) -> BacktestResult:
        """
        Run backtest with attribution analysis.

        Args:
            prices: Price array (should align with features)
            features: Feature array of shape (n_samples, n_features)
            compute_attributions: Whether to compute DeepLift attributions
            compute_baseline: Whether to compute Buy & Hold baseline

        Returns:
            BacktestResult with metrics and trade history
        """
        n = min(len(prices), len(features))
        prices = prices[:n]
        features = features[:n]

        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # -1 (short), 0 (neutral), 1 (long)
        entry_price = 0.0

        results = []
        attribution_history = []
        importance_sum = np.zeros(len(self.feature_names)) if self.feature_names else np.array([])
        n_attributions = 0

        self.model.eval()

        for i in range(n - 1):
            input_tensor = torch.FloatTensor(features[i:i+1])

            # Get prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                predicted_class = torch.argmax(logits, dim=-1).item()

            # Convert class to signal: 0->-1 (Sell), 1->0 (Hold), 2->1 (Buy)
            signal = predicted_class - 1

            # Compute attribution if requested
            top_features = []
            if compute_attributions and self.explainer is not None and self.feature_names:
                try:
                    attribution = self.explainer.get_attributions(
                        input_tensor,
                        feature_names=self.feature_names
                    )
                    top_features = attribution.top_features(5)
                    importance_sum += np.abs(attribution.attributions)
                    n_attributions += 1

                    attribution_history.append({
                        'index': i,
                        'signal': signal,
                        'top_features': top_features,
                        'attributions': attribution.attributions.tolist()
                    })
                except Exception as e:
                    logger.warning(f"Attribution computation failed at step {i}: {e}")

            # Determine new position
            new_position = signal

            # Calculate transaction costs
            position_change = abs(new_position - position)
            cost = 0.0
            if position_change > 0:
                cost = capital * self.transaction_cost * position_change
                cost += capital * self.slippage * position_change

            # Calculate returns
            price_return = (prices[i + 1] - prices[i]) / prices[i]
            position_return = position * price_return

            # Update capital
            old_capital = capital
            capital = capital * (1 + position_return) - cost

            # Track entry for trade analysis
            if position == 0 and new_position != 0:
                entry_price = prices[i]

            # Record result
            result = {
                'index': i,
                'timestamp': i,  # Can be replaced with actual timestamp
                'price': prices[i],
                'signal': signal,
                'position': position,
                'new_position': new_position,
                'price_return': price_return,
                'position_return': position_return,
                'transaction_cost': cost,
                'capital': capital,
                'prob_sell': probs[0],
                'prob_hold': probs[1],
                'prob_buy': probs[2],
            }

            # Add top feature information
            for j, (feat_name, feat_score) in enumerate(top_features[:3]):
                result[f'top_feature_{j+1}'] = feat_name
                result[f'top_score_{j+1}'] = feat_score

            results.append(result)
            position = new_position

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Calculate metrics
        metrics = self._calculate_metrics(results_df)

        # Calculate baseline (Buy & Hold) metrics
        baseline_metrics = None
        if compute_baseline:
            baseline_metrics = self._calculate_baseline_metrics(prices)

        # Average feature importance
        feature_importance = {}
        if n_attributions > 0 and len(self.feature_names) > 0:
            avg_importance = importance_sum / n_attributions
            feature_importance = dict(zip(self.feature_names, avg_importance.tolist()))

        return BacktestResult(
            results_df=results_df,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            feature_importance=feature_importance,
            attribution_history=attribution_history
        )

    def _calculate_metrics(self, df: pd.DataFrame) -> PerformanceMetrics:
        """Calculate performance metrics from backtest results."""
        if len(df) == 0:
            return PerformanceMetrics()

        returns = df['position_return'].values

        # Total return
        final_capital = df['capital'].iloc[-1]
        initial_capital = self.initial_capital
        total_return = (final_capital / initial_capital) - 1

        # Annualized metrics
        n_periods = len(returns)
        ann_factor = self.periods_per_year / max(n_periods, 1)

        ann_return = (1 + total_return) ** ann_factor - 1
        ann_volatility = np.std(returns) * np.sqrt(self.periods_per_year) if np.std(returns) > 0 else 0

        # Sharpe Ratio
        excess_return = np.mean(returns) - self.risk_free_rate / self.periods_per_year
        sharpe = (excess_return / (np.std(returns) + 1e-10)) * np.sqrt(self.periods_per_year)

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino = (excess_return / (downside_std + 1e-10)) * np.sqrt(self.periods_per_year)

        # Maximum Drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Trade analysis
        trades = df[df['position'] != df['new_position'].shift(1)]
        num_trades = len(trades)

        # Win rate
        trade_returns = returns[returns != 0]
        wins = (trade_returns > 0).sum()
        losses = (trade_returns < 0).sum()
        win_rate = wins / max(wins + losses, 1)

        # Average returns
        avg_trade_return = np.mean(trade_returns) if len(trade_returns) > 0 else 0
        avg_win = np.mean(trade_returns[trade_returns > 0]) if wins > 0 else 0
        avg_loss = np.mean(trade_returns[trade_returns < 0]) if losses > 0 else 0

        # Profit factor
        gross_profit = np.sum(trade_returns[trade_returns > 0])
        gross_loss = np.abs(np.sum(trade_returns[trade_returns < 0]))
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_streaks(trade_returns)

        # Calmar Ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            calmar_ratio=calmar
        )

    def _calculate_baseline_metrics(self, prices: np.ndarray) -> PerformanceMetrics:
        """Calculate Buy & Hold baseline metrics."""
        returns = np.diff(prices) / prices[:-1]

        total_return = (prices[-1] / prices[0]) - 1

        n_periods = len(returns)
        ann_factor = self.periods_per_year / max(n_periods, 1)

        ann_return = (1 + total_return) ** ann_factor - 1
        ann_volatility = np.std(returns) * np.sqrt(self.periods_per_year) if np.std(returns) > 0 else 0

        excess_return = np.mean(returns) - self.risk_free_rate / self.periods_per_year
        sharpe = (excess_return / (np.std(returns) + 1e-10)) * np.sqrt(self.periods_per_year)

        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino = (excess_return / (downside_std + 1e-10)) * np.sqrt(self.periods_per_year)

        cumulative = (1 + pd.Series(returns)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()

        wins = (returns > 0).sum()
        losses = (returns < 0).sum()
        win_rate = wins / max(wins + losses, 1)

        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=1,  # Buy & Hold is one trade
            calmar_ratio=calmar
        )

    @staticmethod
    def _calculate_streaks(returns: np.ndarray) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if len(returns) == 0:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for r in returns:
            if r > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif r < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def analyze_attributions_by_outcome(
        self,
        result: BacktestResult
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance segmented by trade outcome.

        Args:
            result: BacktestResult from run_backtest

        Returns:
            Dictionary with 'winning' and 'losing' trade feature importance
        """
        if not result.attribution_history or not self.feature_names:
            return {}

        df = result.results_df
        winning_importance = np.zeros(len(self.feature_names))
        losing_importance = np.zeros(len(self.feature_names))
        n_winning = 0
        n_losing = 0

        for attr_record in result.attribution_history:
            idx = attr_record['index']
            if idx >= len(df):
                continue

            position_return = df.iloc[idx]['position_return']
            attributions = np.array(attr_record['attributions'])

            if position_return > 0:
                winning_importance += np.abs(attributions)
                n_winning += 1
            elif position_return < 0:
                losing_importance += np.abs(attributions)
                n_losing += 1

        result_dict = {}

        if n_winning > 0:
            result_dict['winning'] = dict(zip(
                self.feature_names,
                (winning_importance / n_winning).tolist()
            ))

        if n_losing > 0:
            result_dict['losing'] = dict(zip(
                self.feature_names,
                (losing_importance / n_losing).tolist()
            ))

        return result_dict


def print_backtest_report(result: BacktestResult, show_baseline: bool = True):
    """
    Print a formatted backtest report.

    Args:
        result: BacktestResult from backtest
        show_baseline: Whether to show Buy & Hold comparison
    """
    print("\n" + "=" * 70)
    print("                       BACKTEST REPORT")
    print("=" * 70)

    metrics = result.metrics

    print("\nStrategy Performance:")
    print("-" * 50)
    print(f"  Total Return:            {metrics.total_return * 100:>12.2f}%")
    print(f"  Annualized Return:       {metrics.annualized_return * 100:>12.2f}%")
    print(f"  Annualized Volatility:   {metrics.annualized_volatility * 100:>12.2f}%")
    print(f"  Sharpe Ratio:            {metrics.sharpe_ratio:>12.3f}")
    print(f"  Sortino Ratio:           {metrics.sortino_ratio:>12.3f}")
    print(f"  Calmar Ratio:            {metrics.calmar_ratio:>12.3f}")
    print(f"  Max Drawdown:            {metrics.max_drawdown * 100:>12.2f}%")

    print("\nTrade Statistics:")
    print("-" * 50)
    print(f"  Number of Trades:        {metrics.num_trades:>12}")
    print(f"  Win Rate:                {metrics.win_rate * 100:>12.2f}%")
    print(f"  Profit Factor:           {metrics.profit_factor:>12.3f}")
    print(f"  Avg Trade Return:        {metrics.avg_trade_return * 100:>12.4f}%")
    print(f"  Avg Winning Trade:       {metrics.avg_win * 100:>12.4f}%")
    print(f"  Avg Losing Trade:        {metrics.avg_loss * 100:>12.4f}%")
    print(f"  Max Consecutive Wins:    {metrics.max_consecutive_wins:>12}")
    print(f"  Max Consecutive Losses:  {metrics.max_consecutive_losses:>12}")

    if show_baseline and result.baseline_metrics:
        baseline = result.baseline_metrics
        print("\nBuy & Hold Comparison:")
        print("-" * 50)
        print(f"  {'Metric':<25} {'Strategy':>12} {'Buy&Hold':>12} {'Diff':>10}")
        print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")

        diff_return = metrics.total_return - baseline.total_return
        print(f"  {'Total Return':<25} {metrics.total_return*100:>11.2f}% {baseline.total_return*100:>11.2f}% {diff_return*100:>9.2f}%")

        diff_sharpe = metrics.sharpe_ratio - baseline.sharpe_ratio
        print(f"  {'Sharpe Ratio':<25} {metrics.sharpe_ratio:>12.3f} {baseline.sharpe_ratio:>12.3f} {diff_sharpe:>10.3f}")

        diff_dd = metrics.max_drawdown - baseline.max_drawdown
        print(f"  {'Max Drawdown':<25} {metrics.max_drawdown*100:>11.2f}% {baseline.max_drawdown*100:>11.2f}% {diff_dd*100:>9.2f}%")

    if result.feature_importance:
        print("\nFeature Importance (Average Attribution):")
        print("-" * 50)
        sorted_importance = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (name, score) in enumerate(sorted_importance[:10]):
            bar_len = int(score * 50 / max(result.feature_importance.values()))
            bar = "#" * bar_len
            print(f"  {i+1:2}. {name:<20} {score:.6f} {bar}")

    print("\n" + "=" * 70)


def compare_strategies(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Args:
        results: Dictionary mapping strategy names to BacktestResult

    Returns:
        DataFrame comparing all strategies
    """
    comparison = []

    for name, result in results.items():
        metrics = result.metrics
        row = {
            'Strategy': name,
            'Total Return (%)': metrics.total_return * 100,
            'Ann. Return (%)': metrics.annualized_return * 100,
            'Sharpe': metrics.sharpe_ratio,
            'Sortino': metrics.sortino_ratio,
            'Max DD (%)': metrics.max_drawdown * 100,
            'Win Rate (%)': metrics.win_rate * 100,
            'Profit Factor': metrics.profit_factor,
            'Num Trades': metrics.num_trades,
        }
        comparison.append(row)

    df = pd.DataFrame(comparison)
    df = df.set_index('Strategy')
    return df


if __name__ == "__main__":
    print("Backtesting Framework Demo")
    print("=" * 60)

    # Import dependencies
    from deeplift_model import TradingNetwork, DeepLiftTrading, create_labels_from_returns
    from data_loader import SimulatedDataGenerator, StockDataLoader

    # Generate simulated data
    print("\n1. Generating simulated market data...")
    sim_data = SimulatedDataGenerator.generate_regime_changes(500)
    prices = sim_data['close'].values

    # Prepare features
    print("2. Preparing features...")
    loader = StockDataLoader()
    features, feature_names = loader.prepare_features(sim_data)

    # Align prices with features
    prices = prices[-len(features):]

    # Create labels
    target_horizon = 5
    future_returns = np.zeros(len(prices))
    future_returns[:-target_horizon] = (
        prices[target_horizon:] - prices[:-target_horizon]
    ) / prices[:-target_horizon]

    labels = create_labels_from_returns(future_returns, buy_threshold=0.005, sell_threshold=-0.005)

    # Trim to valid data
    valid_len = len(features) - target_horizon
    X = features[:valid_len]
    y = labels[:valid_len]
    prices_valid = prices[:valid_len + 1]  # +1 for return calculation

    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    prices_test = prices_valid[train_size:]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Create and train model
    print("\n3. Training model...")
    model = TradingNetwork(
        input_size=len(feature_names),
        hidden_sizes=[64, 32],
        num_classes=3,
        dropout_rate=0.2
    )

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

        if epoch % 25 == 0:
            acc = (torch.argmax(outputs, dim=1) == y_train_t).float().mean()
            print(f"   Epoch {epoch:3d}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    # Create explainer
    print("\n4. Setting up DeepLift explainer...")
    explainer = DeepLiftTrading(model)
    explainer.set_baseline_from_data(X_train, method='mean')

    # Run backtest
    print("\n5. Running backtest...")
    backtester = Backtester(
        model=model,
        explainer=explainer,
        feature_names=feature_names,
        transaction_cost=0.001,
        slippage=0.0005,
        initial_capital=10000.0,
        periods_per_year=252
    )

    result = backtester.run_backtest(
        prices_test,
        X_test,
        compute_attributions=True,
        compute_baseline=True
    )

    # Print report
    print_backtest_report(result, show_baseline=True)

    # Analyze attributions by outcome
    print("\n6. Attribution Analysis by Trade Outcome:")
    print("-" * 50)
    outcome_analysis = backtester.analyze_attributions_by_outcome(result)

    if 'winning' in outcome_analysis:
        print("\nTop features in WINNING trades:")
        winning_sorted = sorted(outcome_analysis['winning'].items(), key=lambda x: x[1], reverse=True)
        for name, score in winning_sorted[:5]:
            print(f"  {name}: {score:.6f}")

    if 'losing' in outcome_analysis:
        print("\nTop features in LOSING trades:")
        losing_sorted = sorted(outcome_analysis['losing'].items(), key=lambda x: x[1], reverse=True)
        for name, score in losing_sorted[:5]:
            print(f"  {name}: {score:.6f}")

    print("\nDemo complete!")
