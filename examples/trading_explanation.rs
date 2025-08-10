//! Trading strategy with explanations example.

use deeplift_trading::{
    backtest::BacktestEngine,
    data::{bybit::SimulatedDataGenerator, features::{feature_names, FeatureGenerator}},
    model::TradingNetwork,
    trading::TradingStrategy,
};

fn main() {
    println!("Trading Strategy with DeepLIFT Explanations");
    println!("============================================\n");

    // Generate simulated data with regime changes
    println!("Generating simulated market data with regime changes...");
    let klines = SimulatedDataGenerator::generate_trending_klines(500, 50000.0, 0.02, 0.0001);
    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    println!("Generated {} klines", klines.len());
    println!("Price range: {:.2} - {:.2}",
             prices.iter().cloned().fold(f64::INFINITY, f64::min),
             prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Compute features
    println!("\nComputing technical features...");
    let feature_gen = FeatureGenerator::new(20);
    let features = feature_gen.compute_features(&klines);
    println!("Computed {} feature vectors", features.len());

    // Align prices with features
    let offset = prices.len() - features.len();
    let aligned_prices = &prices[offset..];

    // Create trading network
    println!("\nCreating trading network...");
    let network = TradingNetwork::new(vec![11, 64, 32, 1]);

    // Create strategy with mean reference
    let reference: Vec<f64> = features
        .iter()
        .fold(vec![0.0; 11], |mut acc, f| {
            for (i, val) in f.iter().enumerate() {
                acc[i] += val;
            }
            acc
        })
        .iter()
        .map(|sum| sum / features.len() as f64)
        .collect();

    let strategy = TradingStrategy::new(
        network,
        reference,
        feature_names(),
        0.001, // threshold
    );

    // Run backtest
    println!("\nRunning backtest with explanations...");
    let engine = BacktestEngine::new(strategy, 0.001); // 0.1% transaction cost
    let results = engine.run(&features, aligned_prices, 10000.0);

    // Print metrics
    println!("\n{}", results.metrics);

    // Analyze some trading decisions
    println!("\nSample Trading Decisions with Explanations:");
    println!("-------------------------------------------");

    let sample_indices = [10, 50, 100, 200, results.entries.len() - 10];
    for &i in &sample_indices {
        if i < results.entries.len() {
            let entry = &results.entries[i];
            println!("\nIndex {}: Price = ${:.2}", i, entry.price);
            println!("  Signal: {} (prediction: {:.6})", entry.signal, entry.prediction);
            println!("  Position: {}", match entry.position {
                1 => "Long",
                -1 => "Short",
                _ => "Neutral",
            });
            println!("  Capital: ${:.2}", entry.capital);
            println!("  Top contributing features:");
            for (name, score) in &entry.top_features {
                let direction = if *score > 0.0 { "bullish" } else { "bearish" };
                println!("    {:<20} {:+.6} ({})", name, score, direction);
            }
        }
    }

    // Feature importance throughout the backtest
    println!("\n\nFeature Importance Summary:");
    println!("---------------------------");

    let mut feature_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut feature_total_scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for entry in &results.entries {
        for (name, score) in &entry.top_features {
            *feature_counts.entry(name.clone()).or_insert(0) += 1;
            *feature_total_scores.entry(name.clone()).or_insert(0.0) += score.abs();
        }
    }

    let mut importance: Vec<_> = feature_counts
        .iter()
        .map(|(name, count)| {
            let avg_score = feature_total_scores.get(name).unwrap_or(&0.0) / *count as f64;
            (name.clone(), *count, avg_score)
        })
        .collect();
    importance.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Features appearing in top-3 most frequently:");
    for (name, count, avg_score) in &importance {
        let pct = *count as f64 / results.entries.len() as f64 * 100.0;
        println!("  {:<20} {:>4} times ({:>5.1}%), avg |score|: {:.6}",
                 name, count, pct, avg_score);
    }

    println!("\n[Example completed successfully]");
}
