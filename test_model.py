#!/usr/bin/env python3
"""
Simple BERT Model Testing Script

This script provides an easy way to test your trained BERT models with sample texts
or your own custom inputs.

Usage:
    python test_model.py
    python test_model.py --model ./models_bert_minimal
    python test_model.py --custom
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from evaluate_model import ModelEvaluator
except ImportError:
    print(
        "❌ Error: Could not import ModelEvaluator. Make sure evaluate_model.py exists."
    )
    sys.exit(1)


def test_sample_articles(evaluator):
    """Test with predefined sample articles"""

    print("🧪 Testing with sample articles...\n")

    sample_articles = [
        {
            "text": "Scientists at MIT have developed a new solar panel that is 40% more efficient than current models.",
            "expected": "Real News",
            "category": "Science/Technology",
        },
        {
            "text": "BREAKING: Aliens have been secretly living among us for decades, government finally admits!",
            "expected": "Fake News",
            "category": "Conspiracy",
        },
        {
            "text": "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting.",
            "expected": "Real News",
            "category": "Finance",
        },
        {
            "text": "You won't believe this ONE WEIRD TRICK that doctors HATE! Lose 50 pounds instantly!",
            "expected": "Fake News",
            "category": "Clickbait",
        },
        {
            "text": "Local fire department responds to house fire on Main Street, no injuries reported.",
            "expected": "Real News",
            "category": "Local News",
        },
        {
            "text": "SHOCKING: Celebrity reveals they've been dead for 3 years but nobody noticed!",
            "expected": "Fake News",
            "category": "Celebrity Gossip",
        },
        {
            "text": "New study shows that regular exercise can reduce risk of heart disease by up to 30%.",
            "expected": "Real News",
            "category": "Health",
        },
        {
            "text": "URGENT: The world will end tomorrow unless everyone shares this post exactly 10 times!",
            "expected": "Fake News",
            "category": "Social Media Hoax",
        },
    ]

    correct_predictions = 0
    total_predictions = len(sample_articles)

    for i, article in enumerate(sample_articles, 1):
        print(f"📰 Test {i}/{total_predictions}: {article['category']}")
        print(f"Text: {article['text'][:100]}...")

        result = evaluator.predict_single(article["text"], return_probabilities=True)

        # Check if prediction matches expected
        is_correct = result["prediction"] == article["expected"]
        if is_correct:
            correct_predictions += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"

        print(f"Expected: {article['expected']}")
        print(
            f"Predicted: {result['prediction']} ({result['confidence']:.1%} confidence)"
        )
        print(f"Status: {status}")
        print("-" * 80)

    # Summary
    accuracy = correct_predictions / total_predictions
    print(f"\n📊 SUMMARY:")
    print(f"✅ Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"🎯 Accuracy: {accuracy:.1%}")

    if accuracy >= 0.8:
        print("🎉 Excellent performance!")
    elif accuracy >= 0.6:
        print("👍 Good performance!")
    else:
        print("⚠️ Model may need more training or different parameters")


def test_custom_input(evaluator):
    """Test with user's custom input"""

    print("🎯 CUSTOM TEXT TESTING")
    print("=" * 50)
    print("Enter your own news articles to test!")
    print("Type 'quit', 'exit', or 'q' to finish")
    print("=" * 50)

    while True:
        print("\n📰 Enter a news article:")
        user_input = input("> ").strip()

        if user_input.lower() in ["quit", "exit", "q", ""]:
            print("👋 Thanks for testing!")
            break

        # Get prediction
        result = evaluator.predict_single(user_input, return_probabilities=True)

        # Display results with nice formatting
        print("\n" + "=" * 60)
        print("🔮 PREDICTION RESULTS")
        print("=" * 60)

        # Truncate long text for display
        display_text = user_input[:200] + "..." if len(user_input) > 200 else user_input
        print(f"📄 Text: {display_text}")

        # Prediction with emoji
        prediction_emoji = "🚫" if result["prediction"] == "Fake News" else "✅"
        print(f"\n{prediction_emoji} Prediction: **{result['prediction']}**")
        print(f"📊 Confidence: {result['confidence']:.1%}")

        # Probability bars (visual representation)
        real_prob = result["probabilities"]["Real News"]
        fake_prob = result["probabilities"]["Fake News"]

        real_bar = "█" * int(real_prob * 20)
        fake_bar = "█" * int(fake_prob * 20)

        print(f"\n📈 Probabilities:")
        print(f"   Real News: {real_bar:<20} {real_prob:.1%}")
        print(f"   Fake News: {fake_bar:<20} {fake_prob:.1%}")

        # Confidence interpretation with more detail
        confidence = result["confidence"]
        if confidence >= 0.95:
            interpretation = "🎯 Extremely confident - model is very sure"
        elif confidence >= 0.85:
            interpretation = "✅ Very confident - strong indicators present"
        elif confidence >= 0.70:
            interpretation = "👍 Confident - clear patterns detected"
        elif confidence >= 0.60:
            interpretation = "⚠️ Moderate confidence - some uncertainty"
        else:
            interpretation = "❓ Low confidence - model is unsure"

        print(f"\n💡 {interpretation}")
        print("=" * 60)


def test_batch_examples(evaluator):
    """Test multiple examples quickly"""

    quick_tests = [
        "Apple stock reaches all-time high following quarterly earnings report",
        "MIRACLE CURE: Doctors hate this simple trick that cures everything!",
        "Local school district announces new safety protocols for students",
        "SHOCKING: The moon is actually made of cheese, NASA confirms!",
        "Weather forecast predicts rain for the weekend in most areas",
    ]

    print("⚡ QUICK BATCH TEST")
    print("=" * 40)

    for i, text in enumerate(quick_tests, 1):
        result = evaluator.predict_single(text)
        emoji = "🚫" if result["prediction"] == "Fake News" else "✅"
        print(f"{i}. {emoji} {result['prediction']} ({result['confidence']:.0%})")
        print(f"   Text: {text[:60]}...")
        print()


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Simple BERT Model Testing")
    parser.add_argument(
        "--model",
        default="./models_bert_fast",
        help="Path to trained model (default: ./models_bert_fast)",
    )
    parser.add_argument(
        "--custom", action="store_true", help="Run custom input testing only"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick batch test only"
    )
    parser.add_argument(
        "--samples", action="store_true", help="Run sample articles test only"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        print("\n🔍 Available models:")

        # Look for available models
        current_dir = Path(".")
        model_dirs = [
            d for d in current_dir.iterdir() if d.is_dir() and "model" in d.name.lower()
        ]

        if model_dirs:
            for model_dir in model_dirs:
                print(f"  📁 {model_dir}")
            print(f"\n💡 Try: python {sys.argv[0]} --model ./{model_dirs[0].name}")
        else:
            print("  📁 No models found. Train a model first!")
            print("  💡 Run: python train_bert_minimal.py")

        sys.exit(1)

    print("🚀 BERT Model Testing Tool")
    print("=" * 40)
    print(f"📁 Using model: {args.model}")
    print()

    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Run appropriate test based on arguments
    if args.custom:
        test_custom_input(evaluator)
    elif args.quick:
        test_batch_examples(evaluator)
    elif args.samples:
        test_sample_articles(evaluator)
    else:
        # Run all tests by default
        print("🧪 Running comprehensive tests...\n")

        # Quick test first
        test_batch_examples(evaluator)

        print("\n" + "=" * 60 + "\n")

        # Detailed sample test
        test_sample_articles(evaluator)

        print("\n" + "=" * 60 + "\n")

        # Ask if user wants custom testing
        print("🎯 Would you like to test with your own text? (y/n)")
        if input("> ").lower().startswith("y"):
            test_custom_input(evaluator)
        else:
            print("✅ Testing complete!")


if __name__ == "__main__":
    main()
