//! Feature importance analysis example.

use deeplift_trading::{
    deeplift::{compute_feature_importance, AttributionRule, DeepLIFT},
    model::TradingNetwork,
    data::{bybit::SimulatedDataGenerator, features::{feature_names, FeatureGenerator}},
};

fn main() {
    println!("Feature Importance Analysis Example");
    println!("====================================\n");

    // Generate simulated data
    println!("Generating simulated market data...");
    let klines = SimulatedDataGenerator::generate_trending_klines(500, 50000.0, 0.02, 0.0001);
    println!("Generated {} klines", klines.len());

    // Compute features
    println!("Computing technical features...");
    let feature_gen = FeatureGenerator::new(20);
    let features = feature_gen.compute_features(&klines);
    println!("Computed {} feature vectors with {} features each",
             features.len(),
             features.first().map(|f| f.len()).unwrap_or(0));

    // Create network
    let network = TradingNetwork::new(vec![11, 64, 32, 1]);
    println!("\nCreated trading network");

    // Create DeepLIFT explainer
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

    println!("Reference (mean features):");
    for (name, val) in feature_names().iter().zip(reference.iter()) {
        println!("  {:<20} {:.6}", name, val);
    }

    let mut explainer = DeepLIFT::new(reference, AttributionRule::Rescale);
    explainer.set_network(network.weights.clone(), network.biases.clone());

    // Compute feature importance
    println!("\nComputing feature importance across {} samples...", features.len().min(100));
    let names = feature_names();
    let importance = compute_feature_importance(&explainer, &features[..100.min(features.len())], names);

    // Sort by importance
    let mut sorted_importance: Vec<_> = importance.into_iter().collect();
    sorted_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nFeature Importance Ranking:");
    println!("---------------------------");
    for (rank, (name, score)) in sorted_importance.iter().enumerate() {
        let bar_len = (score * 50.0).min(50.0) as usize;
        let bar: String = "=".repeat(bar_len);
        println!("{:>2}. {:<20} {:.6} |{}", rank + 1, name, score, bar);
    }

    // Analyze specific samples
    println!("\n\nSample Analysis:");
    println!("----------------");

    for i in [0, features.len() / 2, features.len() - 1] {
        if i < features.len() {
            let prediction = network.forward(&features[i]);
            let attribution = explainer.attribute(&features[i], feature_names());

            println!("\nSample {}:", i);
            println!("  Prediction: {:.6}", prediction);
            println!("  Top contributors:");
            for (name, score) in attribution.top_features(3) {
                println!("    {:<20} {:+.6}", name, score);
            }
        }
    }

    println!("\n[Example completed successfully]");
}
