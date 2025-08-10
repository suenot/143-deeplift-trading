//! Backtesting engine with DeepLIFT explanations.

use crate::data::bybit::Kline;
use crate::deeplift::Attribution;
use crate::trading::{TradingSignal, TradingStrategy};

/// Single backtest result entry.
#[derive(Debug, Clone)]
pub struct BacktestEntry {
    pub index: usize,
    pub timestamp: i64,
    pub price: f64,
    pub prediction: f64,
    pub signal: TradingSignal,
    pub position: i32,
    pub position_return: f64,
    pub capital: f64,
    pub top_features: Vec<(String, f64)>,
}

/// Backtest results.
#[derive(Debug)]
pub struct BacktestResults {
    pub entries: Vec<BacktestEntry>,
    pub metrics: BacktestMetrics,
}

/// Backtest performance metrics.
#[derive(Debug, Clone, Default)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
}

impl std::fmt::Display for BacktestMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Backtest Metrics:")?;
        writeln!(f, "  Total Return:      {:>10.2}%", self.total_return * 100.0)?;
        writeln!(
            f,
            "  Annualized Return: {:>10.2}%",
            self.annualized_return * 100.0
        )?;
        writeln!(
            f,
            "  Annualized Vol:    {:>10.2}%",
            self.annualized_volatility * 100.0
        )?;
        writeln!(f, "  Sharpe Ratio:      {:>10.3}", self.sharpe_ratio)?;
        writeln!(f, "  Sortino Ratio:     {:>10.3}", self.sortino_ratio)?;
        writeln!(f, "  Max Drawdown:      {:>10.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Win Rate:          {:>10.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Profit Factor:     {:>10.3}", self.profit_factor)?;
        writeln!(f, "  Num Trades:        {:>10}", self.num_trades)
    }
}

/// Backtesting engine with DeepLIFT explanations.
#[derive(Debug)]
pub struct BacktestEngine {
    /// Trading strategy
    pub strategy: TradingStrategy,
    /// Transaction cost as fraction
    pub transaction_cost: f64,
    /// Number of top features to track
    pub n_top_features: usize,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(strategy: TradingStrategy, transaction_cost: f64) -> Self {
        Self {
            strategy,
            transaction_cost,
            n_top_features: 3,
        }
    }

    /// Run backtest on features and prices.
    pub fn run(
        &self,
        features: &[Vec<f64>],
        prices: &[f64],
        initial_capital: f64,
    ) -> BacktestResults {
        let n = features.len().min(prices.len());
        let mut entries = Vec::with_capacity(n);
        let mut capital = initial_capital;
        let mut position = 0;

        for i in 0..n {
            let prediction = self.strategy.model.forward(&features[i]);
            let (signal, attribution) = self
                .strategy
                .generate_signal_with_explanation(&features[i]);
            let new_position = signal.position();

            // Transaction costs
            if new_position != position && i > 0 {
                capital *= 1.0 - self.transaction_cost;
            }

            // Calculate returns
            let (actual_return, position_return) = if i < n - 1 {
                let ret = prices[i + 1] / prices[i] - 1.0;
                let pos_ret = position as f64 * ret;
                capital *= 1.0 + pos_ret;
                (ret, pos_ret)
            } else {
                (0.0, 0.0)
            };

            entries.push(BacktestEntry {
                index: i,
                timestamp: 0, // Would come from klines
                price: prices[i],
                prediction,
                signal,
                position,
                position_return,
                capital,
                top_features: attribution.top_features(self.n_top_features),
            });

            position = new_position;
        }

        let metrics = self.calculate_metrics(&entries, initial_capital);

        BacktestResults { entries, metrics }
    }

    /// Run backtest on klines.
    pub fn run_on_klines(
        &self,
        klines: &[Kline],
        features: &[Vec<f64>],
        initial_capital: f64,
    ) -> BacktestResults {
        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let offset = prices.len().saturating_sub(features.len());
        let aligned_prices = &prices[offset..];

        self.run(features, aligned_prices, initial_capital)
    }

    /// Calculate backtest metrics.
    fn calculate_metrics(&self, entries: &[BacktestEntry], initial_capital: f64) -> BacktestMetrics {
        if entries.is_empty() {
            return BacktestMetrics::default();
        }

        let returns: Vec<f64> = entries.iter().map(|e| e.position_return).collect();
        let n = returns.len() as f64;

        // Total return
        let final_capital = entries.last().map(|e| e.capital).unwrap_or(initial_capital);
        let total_return = final_capital / initial_capital - 1.0;

        // Annualized (assuming hourly data)
        let periods_per_year = 8760.0;
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n) - 1.0;

        // Volatility
        let mean_return: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let annualized_volatility = std * periods_per_year.sqrt();

        // Sharpe ratio
        let sharpe_ratio = if std > 0.0 {
            periods_per_year.sqrt() * mean_return / std
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let sortino_ratio = if !downside_returns.is_empty() {
            let downside_var: f64 =
                downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
            let downside_std = downside_var.sqrt();
            if downside_std > 0.0 {
                periods_per_year.sqrt() * mean_return / downside_std
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

        // Max drawdown
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        for r in &returns {
            cumulative *= 1.0 + r;
            peak = peak.max(cumulative);
            let drawdown = cumulative / peak - 1.0;
            max_drawdown = max_drawdown.min(drawdown);
        }

        // Win rate
        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        let losses = returns.iter().filter(|&&r| r < 0.0).count();
        let win_rate = if wins + losses > 0 {
            wins as f64 / (wins + losses) as f64
        } else {
            0.0
        };

        // Profit factor
        let gross_profits: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_losses > 0.0 {
            gross_profits / gross_losses
        } else if gross_profits > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Number of trades
        let num_trades = entries
            .windows(2)
            .filter(|w| w[0].position != w[1].position)
            .count();

        BacktestMetrics {
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::features::feature_names;
    use crate::model::TradingNetwork;

    #[test]
    fn test_backtest_engine() {
        let model = TradingNetwork::new(vec![11, 32, 1]);
        let reference = vec![0.0; 11];
        let names = feature_names();
        let strategy = TradingStrategy::new(model, reference, names, 0.001);

        let engine = BacktestEngine::new(strategy, 0.001);

        // Generate dummy data
        let features: Vec<Vec<f64>> = (0..100)
            .map(|_| (0..11).map(|_| rand::random::<f64>() * 0.1).collect())
            .collect();
        let prices: Vec<f64> = (0..100).map(|i| 50000.0 + i as f64 * 10.0).collect();

        let results = engine.run(&features, &prices, 10000.0);

        assert_eq!(results.entries.len(), 100);
        assert!(results.metrics.total_return.is_finite());
    }
}
