//! # Trading Strategy
//!
//! DeepLift-based trading strategy with position management.

use crate::{DeepLiftError, DeepLiftNetwork, Result, TradingSignal, SignalGenerator};
use serde::{Deserialize, Serialize};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    /// No position
    Flat,
    /// Long position
    Long,
    /// Short position
    Short,
}

impl std::fmt::Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSide::Flat => write!(f, "FLAT"),
            PositionSide::Long => write!(f, "LONG"),
            PositionSide::Short => write!(f, "SHORT"),
        }
    }
}

/// Current position state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Position side
    pub side: PositionSide,
    /// Entry price
    pub entry_price: f64,
    /// Position size (quantity)
    pub size: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Entry timestamp
    pub entry_time: u64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            side: PositionSide::Flat,
            entry_price: 0.0,
            size: 0.0,
            unrealized_pnl: 0.0,
            entry_time: 0,
        }
    }
}

impl Position {
    /// Create a new flat position
    pub fn flat() -> Self {
        Self::default()
    }

    /// Create a long position
    pub fn long(entry_price: f64, size: f64, entry_time: u64) -> Self {
        Self {
            side: PositionSide::Long,
            entry_price,
            size,
            unrealized_pnl: 0.0,
            entry_time,
        }
    }

    /// Create a short position
    pub fn short(entry_price: f64, size: f64, entry_time: u64) -> Self {
        Self {
            side: PositionSide::Short,
            entry_price,
            size,
            unrealized_pnl: 0.0,
            entry_time,
        }
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        !matches!(self.side, PositionSide::Flat)
    }

    /// Update unrealized PnL based on current price
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = match self.side {
            PositionSide::Long => (current_price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - current_price) * self.size,
            PositionSide::Flat => 0.0,
        };
    }

    /// Get return percentage
    pub fn return_pct(&self, current_price: f64) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        match self.side {
            PositionSide::Long => (current_price - self.entry_price) / self.entry_price * 100.0,
            PositionSide::Short => (self.entry_price - current_price) / self.entry_price * 100.0,
            PositionSide::Flat => 0.0,
        }
    }
}

/// Trade action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    /// Open long position
    OpenLong { size: f64 },
    /// Open short position
    OpenShort { size: f64 },
    /// Close current position
    Close,
    /// No action
    Hold,
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Position size as fraction of capital
    pub position_size: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Minimum confidence to enter trade
    pub min_confidence: f64,
    /// Allow short positions
    pub allow_short: bool,
    /// Maximum holding period (in bars)
    pub max_holding_period: Option<usize>,
    /// Trailing stop percentage (if enabled)
    pub trailing_stop: Option<f64>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            position_size: 0.1,
            stop_loss: 0.02,
            take_profit: 0.04,
            min_confidence: 0.3,
            allow_short: true,
            max_holding_period: None,
            trailing_stop: None,
        }
    }
}

/// DeepLift-based trading strategy
pub struct DeepLiftStrategy {
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Strategy configuration
    config: StrategyConfig,
    /// Current position
    position: Position,
    /// Available capital
    capital: f64,
    /// Total realized PnL
    realized_pnl: f64,
    /// Trade count
    trade_count: usize,
    /// Winning trades
    winning_trades: usize,
    /// Current bar index
    current_bar: usize,
    /// Entry bar index
    entry_bar: Option<usize>,
    /// Highest price since entry (for trailing stop)
    highest_since_entry: f64,
    /// Lowest price since entry (for trailing stop)
    lowest_since_entry: f64,
}

impl DeepLiftStrategy {
    /// Create a new strategy
    pub fn new(network: DeepLiftNetwork, capital: f64) -> Self {
        let signal_generator = SignalGenerator::new(network);
        Self {
            signal_generator,
            config: StrategyConfig::default(),
            position: Position::flat(),
            capital,
            realized_pnl: 0.0,
            trade_count: 0,
            winning_trades: 0,
            current_bar: 0,
            entry_bar: None,
            highest_since_entry: 0.0,
            lowest_since_entry: f64::MAX,
        }
    }

    /// Set strategy configuration
    pub fn with_config(mut self, config: StrategyConfig) -> Self {
        self.config = config;
        self
    }

    /// Process a new bar and determine action
    pub fn on_bar(
        &mut self,
        features: &[f64],
        current_price: f64,
        timestamp: u64,
    ) -> Result<TradeAction> {
        self.current_bar += 1;

        // Update tracking for trailing stop
        if self.position.is_open() {
            self.highest_since_entry = self.highest_since_entry.max(current_price);
            self.lowest_since_entry = self.lowest_since_entry.min(current_price);
        }

        // Check exit conditions first if in position
        if self.position.is_open() {
            if let Some(action) = self.check_exit_conditions(current_price)? {
                return Ok(action);
            }
        }

        // Get signal and explanation
        let explanation = self.signal_generator.get_signal_explanation(features)?;

        // Determine action based on signal and current position
        let action = self.determine_action(&explanation.signal, explanation.confidence, current_price, timestamp)?;

        Ok(action)
    }

    /// Check exit conditions (stop loss, take profit, etc.)
    fn check_exit_conditions(&self, current_price: f64) -> Result<Option<TradeAction>> {
        let return_pct = self.position.return_pct(current_price);

        // Check stop loss
        if return_pct <= -self.config.stop_loss * 100.0 {
            return Ok(Some(TradeAction::Close));
        }

        // Check take profit
        if return_pct >= self.config.take_profit * 100.0 {
            return Ok(Some(TradeAction::Close));
        }

        // Check trailing stop
        if let Some(trailing) = self.config.trailing_stop {
            match self.position.side {
                PositionSide::Long => {
                    let trailing_stop_price = self.highest_since_entry * (1.0 - trailing);
                    if current_price <= trailing_stop_price {
                        return Ok(Some(TradeAction::Close));
                    }
                }
                PositionSide::Short => {
                    let trailing_stop_price = self.lowest_since_entry * (1.0 + trailing);
                    if current_price >= trailing_stop_price {
                        return Ok(Some(TradeAction::Close));
                    }
                }
                PositionSide::Flat => {}
            }
        }

        // Check max holding period
        if let (Some(max_period), Some(entry_bar)) = (self.config.max_holding_period, self.entry_bar) {
            if self.current_bar - entry_bar >= max_period {
                return Ok(Some(TradeAction::Close));
            }
        }

        Ok(None)
    }

    /// Determine action based on signal
    fn determine_action(
        &mut self,
        signal: &TradingSignal,
        confidence: f64,
        current_price: f64,
        timestamp: u64,
    ) -> Result<TradeAction> {
        // Check minimum confidence
        if confidence < self.config.min_confidence && !self.position.is_open() {
            return Ok(TradeAction::Hold);
        }

        match (&self.position.side, signal) {
            // Flat position
            (PositionSide::Flat, TradingSignal::Buy) => {
                let size = (self.capital * self.config.position_size) / current_price;
                self.open_position(PositionSide::Long, current_price, size, timestamp);
                Ok(TradeAction::OpenLong { size })
            }
            (PositionSide::Flat, TradingSignal::Sell) if self.config.allow_short => {
                let size = (self.capital * self.config.position_size) / current_price;
                self.open_position(PositionSide::Short, current_price, size, timestamp);
                Ok(TradeAction::OpenShort { size })
            }
            
            // Long position
            (PositionSide::Long, TradingSignal::Sell) => {
                self.close_position(current_price);
                Ok(TradeAction::Close)
            }
            
            // Short position
            (PositionSide::Short, TradingSignal::Buy) => {
                self.close_position(current_price);
                Ok(TradeAction::Close)
            }
            
            // Otherwise hold
            _ => Ok(TradeAction::Hold),
        }
    }

    /// Open a new position
    fn open_position(&mut self, side: PositionSide, price: f64, size: f64, timestamp: u64) {
        self.position = match side {
            PositionSide::Long => Position::long(price, size, timestamp),
            PositionSide::Short => Position::short(price, size, timestamp),
            PositionSide::Flat => Position::flat(),
        };
        self.entry_bar = Some(self.current_bar);
        self.highest_since_entry = price;
        self.lowest_since_entry = price;
    }

    /// Close current position
    fn close_position(&mut self, price: f64) {
        if !self.position.is_open() {
            return;
        }

        // Calculate realized PnL
        self.position.update_pnl(price);
        let pnl = self.position.unrealized_pnl;
        self.realized_pnl += pnl;
        self.capital += pnl;
        
        // Update statistics
        self.trade_count += 1;
        if pnl > 0.0 {
            self.winning_trades += 1;
        }

        // Reset position
        self.position = Position::flat();
        self.entry_bar = None;
    }

    /// Get current position
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Get capital
    pub fn capital(&self) -> f64 {
        self.capital
    }

    /// Get realized PnL
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get total PnL (realized + unrealized)
    pub fn total_pnl(&self, current_price: f64) -> f64 {
        let mut pos = self.position.clone();
        pos.update_pnl(current_price);
        self.realized_pnl + pos.unrealized_pnl
    }

    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        if self.trade_count == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / self.trade_count as f64
    }

    /// Get trade count
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }

    /// Get signal generator reference
    pub fn signal_generator(&self) -> &SignalGenerator {
        &self.signal_generator
    }

    /// Get mutable signal generator reference
    pub fn signal_generator_mut(&mut self) -> &mut SignalGenerator {
        &mut self.signal_generator
    }

    /// Reset strategy state (keep network weights)
    pub fn reset(&mut self, capital: f64) {
        self.position = Position::flat();
        self.capital = capital;
        self.realized_pnl = 0.0;
        self.trade_count = 0;
        self.winning_trades = 0;
        self.current_bar = 0;
        self.entry_bar = None;
        self.highest_since_entry = 0.0;
        self.lowest_since_entry = f64::MAX;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeepLiftConfig;

    #[test]
    fn test_position_flat() {
        let pos = Position::flat();
        assert!(!pos.is_open());
        assert_eq!(pos.side, PositionSide::Flat);
    }

    #[test]
    fn test_position_long() {
        let pos = Position::long(100.0, 1.0, 12345);
        assert!(pos.is_open());
        assert_eq!(pos.side, PositionSide::Long);
        assert_eq!(pos.entry_price, 100.0);
    }

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::long(100.0, 1.0, 12345);
        pos.update_pnl(110.0);
        assert_eq!(pos.unrealized_pnl, 10.0);
        
        let mut pos_short = Position::short(100.0, 1.0, 12345);
        pos_short.update_pnl(90.0);
        assert_eq!(pos_short.unrealized_pnl, 10.0);
    }

    #[test]
    fn test_position_return_pct() {
        let pos = Position::long(100.0, 1.0, 12345);
        assert_eq!(pos.return_pct(110.0), 10.0);
        
        let pos_short = Position::short(100.0, 1.0, 12345);
        assert_eq!(pos_short.return_pct(90.0), 10.0);
    }

    #[test]
    fn test_strategy_creation() {
        let config = DeepLiftConfig::new(10, vec![20, 10], 1);
        let network = DeepLiftNetwork::new(config);
        let strategy = DeepLiftStrategy::new(network, 10000.0);
        assert_eq!(strategy.capital(), 10000.0);
        assert!(!strategy.position().is_open());
    }

    #[test]
    fn test_strategy_on_bar() {
        let config = DeepLiftConfig::new(5, vec![10], 1);
        let network = DeepLiftNetwork::new(config);
        let mut strategy = DeepLiftStrategy::new(network, 10000.0);
        
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = strategy.on_bar(&features, 100.0, 12345);
        assert!(result.is_ok());
    }

    #[test]
    fn test_win_rate() {
        let config = DeepLiftConfig::new(5, vec![10], 1);
        let network = DeepLiftNetwork::new(config);
        let strategy = DeepLiftStrategy::new(network, 10000.0);
        assert_eq!(strategy.win_rate(), 0.0);
    }

    #[test]
    fn test_strategy_reset() {
        let config = DeepLiftConfig::new(5, vec![10], 1);
        let network = DeepLiftNetwork::new(config);
        let mut strategy = DeepLiftStrategy::new(network, 10000.0);
        strategy.reset(20000.0);
        assert_eq!(strategy.capital(), 20000.0);
    }
}
