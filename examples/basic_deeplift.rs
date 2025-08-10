//! Basic DeepLIFT attribution example.

use deeplift_trading::{
    deeplift::{Attribution, AttributionRule, DeepLIFT},
    model::TradingNetwork,
    data::features::feature_names,
};

fn main() {
    println!("DeepLIFT Attribution Example");
    println!("============================\n");

    // Create a simple trading network
    let network = TradingNetwork::new(vec![11, 32, 16, 1]);
    println!("Created network with layers: [11, 32, 16, 1]");

    // Create DeepLIFT explainer with zero reference
    let reference = vec![0.0; 11];
    let mut explainer = DeepLIFT::new(reference, AttributionRule::Rescale);
    explainer.set_network(network.weights.clone(), network.biases.clone());

    // Sample input features
    let features = vec![
        0.02,   // returns_1d
        0.05,   // returns_5d
        0.08,   // returns_10d
        0.01,   // sma_ratio
        0.015,  // ema_ratio
        0.02,   // volatility
        0.03,   // momentum
        0.35,   // rsi (normalized)
        0.002,  // macd
        -0.5,   // bb_position
        0.2,    // volume_sma_ratio
    ];

    // Get prediction
    let prediction = network.forward(&features);
    println!("Prediction: {:.6}", prediction);

    // Compute attribution
    let names = feature_names();
    let attribution = explainer.attribute(&features, names);

    println!("\nAttribution Results:");
    println!("-------------------");
    println!("Baseline output: {:.6}", attribution.baseline_output);
    println!("Actual output:   {:.6}", attribution.actual_output);
    println!("Delta:           {:.6}", attribution.delta);
    println!("Summation error: {:.10}", attribution.verify_summation());

    println!("\nTop 5 Contributing Features:");
    for (name, score) in attribution.top_features(5) {
        let direction = if score > 0.0 { "+" } else { "" };
        println!("  {:<20} {}{:.6}", name, direction, score);
    }

    println!("\nPositive Contributors:");
    for (name, score) in attribution.positive_contributors() {
        println!("  {:<20} +{:.6}", name, score);
    }

    println!("\nNegative Contributors:");
    for (name, score) in attribution.negative_contributors() {
        println!("  {:<20} {:.6}", name, score);
    }

    println!("\n[Example completed successfully]");
}
