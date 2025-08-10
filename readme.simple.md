# DeepLift Trading - Simple Explanation

## What is DeepLift?

Imagine you have a very smart friend who can predict whether a stock will go up or down. You ask them, "Should I buy Apple stock?" They say "Yes, buy it!" But when you ask "Why?", they just shrug and say "I just know."

That is frustrating, right? You want to understand WHY before risking your money.

**DeepLift is like giving your smart friend the ability to explain their reasoning.**

### The Detective Analogy

Think of DeepLift as a detective investigating a crime scene:

**The Case:** The AI predicted "BUY this stock"

**Without DeepLift (Bad Detective):**
- "I think you should buy."
- "Why?"
- "I cannot explain. Trust me."

**With DeepLift (Good Detective):**
- "I think you should buy."
- "Why?"
- "Here are the clues I found:"
  - "The price dropped 10% but started recovering (35% of my decision)"
  - "Trading volume suddenly increased (25% of my decision)"
  - "The RSI indicator shows oversold conditions (20% of my decision)"
  - "Other smaller factors (20% of my decision)"

Now you can verify if the reasoning makes sense before making your trade!

---

## Why Use DeepLift in Trading?

### The Black Box Problem

When you train an AI to predict stock movements, it becomes a mystery box:

```
[Stock Data] --> [AI Brain] --> "BUY!"
                    ???
            What is happening inside?
```

This is risky because:

1. **Trust Issues** - How do you know the AI is not just guessing?
2. **Debugging** - If the AI loses money, how do you fix it?
3. **Regulations** - Banks must explain their trading decisions
4. **Learning** - You want to learn from what the AI discovered!

### Real Trading Scenario

**Situation:** Your AI model says "Buy Tesla stock with high confidence"

**Without DeepLift:**
You blindly follow the signal and hope for the best.

**With DeepLift:**
You see the breakdown:
- Price momentum: +40% contribution
- Volume spike: +30% contribution
- Day of week is Friday: +25% contribution
- Other factors: +5% contribution

**Wait!** Why is "Friday" contributing so much? That seems suspicious. Maybe the AI learned a random pattern that will not work in the future. Time to investigate before trading!

---

## How Does It Work?

DeepLift uses a clever trick: it compares the current situation to a "neutral" baseline.

### Step 1: Define What "Normal" Looks Like

First, we pick a reference point. Think of it as "a boring day with nothing special happening."

```
Reference (Neutral Day):
- Price change: 0%
- Volume: Average
- RSI: 50 (neutral)
- Everything else: Normal values
```

### Step 2: See How Today Differs from Normal

```
Today's Data:
- Price change: -5% (dropped!)
- Volume: 2x higher than average
- RSI: 25 (very oversold)
```

### Step 3: Track the Differences Through the AI

DeepLift follows each difference through every layer of the neural network.

```
Input Differences --> Layer 1 --> Layer 2 --> Final Prediction
     |                  |           |              |
     v                  v           v              v
  "How much        "Process     "Process      "This much
   did each         them"        more"         came from
   feature                                     each input"
   change?"
```

### Step 4: Get the Contribution Scores

At the end, DeepLift tells you exactly how much each input contributed:

```
Final Prediction: 0.75 (Strong Buy Signal)

Contributions:
- Price drop (-5%): +0.30 (helped the buy signal)
- High volume: +0.25 (confirmed the signal)
- Low RSI (25): +0.15 (indicated oversold)
- Other factors: +0.05

Total: 0.30 + 0.25 + 0.15 + 0.05 = 0.75 (Matches!)
```

The magic: **All contributions add up exactly to the prediction!**

---

## Quick Start Guide

Here is a simple Python example you can run:

```python
import torch
from deeplift_trader import DeepLIFT, TradingModelWithDeepLIFT

# Step 1: Create a simple trading model
model = TradingModelWithDeepLIFT(input_size=5)

# Step 2: Define your feature names
features = ["price_change", "volume", "rsi", "momentum", "volatility"]

# Step 3: Create sample data (today's market conditions)
today = torch.tensor([[0.02, 1.5, 0.3, 0.05, 0.01]])

# Step 4: Set up DeepLift with a neutral baseline (zeros)
explainer = DeepLIFT(model, reference=torch.zeros(1, 5))

# Step 5: Get the explanation
result = explainer.attribute(today, features)

# Step 6: See what drove the prediction
print(f"Prediction: {result.actual_output:.2f}")
print("Top factors:")
for name, score in result.top_features(3):
    print(f"  {name}: {score:.3f}")
```

**What each part does:**
- Lines 1-2: Import the tools we need
- Line 5: Create an AI model that takes 5 inputs
- Line 8: Name our inputs so we can understand them
- Line 11: Create today's market data
- Line 14: Set up DeepLift (zeros = neutral baseline)
- Line 17: Ask DeepLift to explain the prediction
- Lines 20-23: Print which features mattered most

---

## Real-World Trading Example

### Why Did the Model Say BUY for Apple Stock?

Let us walk through a concrete example:

**The Situation:**
- Date: January 15th
- Stock: Apple (AAPL)
- AI Prediction: BUY with score 0.82

**Input Features:**
| Feature | Value | Compared to Normal |
|---------|-------|-------------------|
| 5-day return | -8% | Much lower |
| Volume | 2.3x average | Much higher |
| RSI | 28 | Oversold |
| Price vs Moving Avg | -5% | Below average |
| Volatility | 0.03 | Slightly high |

**DeepLift Explanation:**

```
Why the model said BUY (score: 0.82):

1. RSI at 28 (oversold)      --> +0.32 (39%)
   "The stock is beaten down and likely to bounce"

2. 5-day return of -8%       --> +0.25 (30%)
   "Big recent drop creates buying opportunity"

3. Volume 2.3x higher        --> +0.15 (18%)
   "High volume suggests institutional buying"

4. Price below moving avg    --> +0.08 (10%)
   "Stock is cheap relative to recent prices"

5. Higher volatility         --> +0.02 (3%)
   "Minor positive signal"

Total contribution: 0.32 + 0.25 + 0.15 + 0.08 + 0.02 = 0.82
```

**Interpretation:**
The AI sees a classic "oversold bounce" setup. The stock dropped significantly, RSI hit oversold levels, and big volume suggests buyers are stepping in. This matches what experienced traders look for!

---

## Key Takeaways

- **DeepLift explains AI decisions** by breaking down which inputs contributed to each prediction

- **It compares to a baseline** - you choose what "normal" looks like (usually zeros or averages)

- **Contributions always sum up** - if the model predicts 0.75, the feature contributions add up to exactly 0.75

- **Use it to build trust** - verify the AI is using sensible features, not random patterns

- **Debug your models** - find suspicious features that might cause future failures

- **Learn market insights** - discover which indicators matter most in different conditions

- **Meet regulations** - explain trading decisions to auditors and risk managers

---

## What's Next?

Now that you understand DeepLift, you can:

1. **Run the examples** in this chapter to see DeepLift in action

2. **Train your own model** on real stock or crypto data

3. **Analyze feature importance** to see which indicators drive your strategy

4. **Compare market regimes** - see how feature importance shifts in bull vs bear markets

5. **Explore related methods** - check out SHAP and Integrated Gradients for alternative explanations

**Remember:** An AI that can explain itself is an AI you can trust, improve, and rely on for real trading decisions.

---

*Previous: [Chapter 121: Layer-wise Relevance Propagation](../121_layer_wise_relevance)*

*Next: [Chapter 123: GradCAM for Finance](../123_gradcam_finance)*
