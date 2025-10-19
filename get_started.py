#!/usr/bin/env python3
"""
BERT Fake News Detection - Getting Started Script

This script provides a simple way for new users to get started with the project.
It will guide them through the setup and run a quick demo.

Usage:
    python get_started.py
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("🎉 Welcome to BERT Fake News Detection!")
    print("=" * 50)
    print("This script will help you get started quickly.")
    print("=" * 50)


def check_requirements():
    """Check if requirements are installed"""
    print("\n1️⃣ Checking requirements...")

    try:
        import torch
        import transformers

        print("✅ Core packages found!")
        return True
    except ImportError:
        print("⚠️  Missing packages. Installing now...")
        return install_requirements()


def install_requirements():
    """Install requirements"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        print("💡 Try running: pip install -r requirements.txt")
        return False


def run_quick_training():
    """Run quick training"""
    print("\n2️⃣ Running quick training (1-2 minutes)...")
    print("🏋️‍♀️ Training a small BERT model on sample data...")

    try:
        result = subprocess.run(
            [sys.executable, "train_bert_minimal.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print("⚠️  Training had issues, but let's continue...")
            print(f"Error: {result.stderr[-200:]}")  # Show last 200 chars
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  Training is taking longer than expected...")
        print("💡 You can run 'python train_bert_minimal.py' manually later")
        return False
    except Exception as e:
        print(f"⚠️  Training error: {e}")
        return False


def run_quick_test():
    """Run quick test"""
    print("\n3️⃣ Testing the model...")

    # Check if model exists
    model_path = Path("models_bert_minimal")
    if not model_path.exists():
        print("⚠️  No trained model found. Let's test with sample predictions...")
        show_sample_predictions()
        return False

    try:
        result = subprocess.run(
            [sys.executable, "test_model.py", "--quick"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print("✅ Model testing successful!")
            print("\n🎯 Test Results:")
            print(result.stdout[-500:])  # Show last 500 chars
            return True
        else:
            print("⚠️  Testing had issues...")
            show_sample_predictions()
            return False

    except Exception as e:
        print(f"⚠️  Testing error: {e}")
        show_sample_predictions()
        return False


def show_sample_predictions():
    """Show what predictions would look like"""
    print("\n📰 Here's what the predictions look like:")
    print("-" * 40)

    samples = [
        (
            "Real news: The Federal Reserve announced interest rate changes",
            "Real News",
            "85%",
        ),
        (
            "Fake news: SHOCKING! Aliens confirm they built pyramids!",
            "Fake News",
            "95%",
        ),
        (
            "Real news: Local fire department responds to emergency call",
            "Real News",
            "78%",
        ),
        ("Fake news: You won't believe this ONE WEIRD TRICK!", "Fake News", "92%"),
    ]

    for text, prediction, confidence in samples:
        emoji = "✅" if prediction == "Real News" else "🚫"
        print(f"{emoji} {prediction} ({confidence}) - {text[:50]}...")


def show_next_steps():
    """Show what users can do next"""
    print("\n🚀 What you can do next:")
    print("-" * 30)
    print("📖 Interactive Testing:")
    print(
        "   python evaluate_model.py --model_path ./models_bert_minimal --interactive"
    )
    print("")
    print("🎯 Test single article:")
    print(
        '   python evaluate_model.py --model_path ./models_bert_minimal --text "Your article here"'
    )
    print("")
    print("⚡ Quick batch test:")
    print("   python test_model.py --quick")
    print("")
    print("🚀 Better training (5-10 min):")
    print("   python train_bert_fast.py")
    print("")
    print("☁️  GPU Training (Google Colab):")
    print("   Upload BERT_Training_Colab.ipynb to Google Colab")
    print("")
    print("📚 Read the full README.md for detailed instructions!")


def run_interactive_demo():
    """Run a simple interactive demo"""
    print("\n🎮 Interactive Demo")
    print("-" * 20)
    print("Let's test some sample articles!")

    samples = [
        "Scientists at MIT develop revolutionary solar panel technology",
        "BREAKING: Chocolate now proven to be a vegetable by experts!",
        "Stock market closes mixed following economic data release",
        "You won't believe this MIRACLE CURE that doctors hate!",
    ]

    print("\nSample articles to test:")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")

    try:
        choice = input("\nEnter number (1-4) to test, or press Enter to skip: ").strip()

        if choice in ["1", "2", "3", "4"]:
            article = samples[int(choice) - 1]
            print(f"\n📰 Testing: {article}")

            # Try to run actual prediction
            model_path = Path("models_bert_minimal")
            if model_path.exists():
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "evaluate_model.py",
                            "--model_path",
                            "./models_bert_minimal",
                            "--text",
                            article,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if result.returncode == 0:
                        print(result.stdout)
                    else:
                        show_mock_prediction(article)
                except:
                    show_mock_prediction(article)
            else:
                show_mock_prediction(article)

    except KeyboardInterrupt:
        print("\n👋 Demo cancelled!")
    except:
        print("\n⚠️  Demo error, but that's okay!")


def show_mock_prediction(article):
    """Show a mock prediction"""
    if "BREAKING" in article or "MIRACLE" in article or "chocolate" in article.lower():
        prediction = "Fake News"
        confidence = "94%"
        emoji = "🚫"
    else:
        prediction = "Real News"
        confidence = "87%"
        emoji = "✅"

    print(f"\n{emoji} Prediction: {prediction}")
    print(f"📊 Confidence: {confidence}")
    print("(This is a mock prediction - train a model for real results!)")


def main():
    """Main function"""
    print_banner()

    # Step 1: Check/install requirements
    if not check_requirements():
        print("\n❌ Cannot continue without required packages")
        print("💡 Please run: pip install -r requirements.txt")
        return

    # Step 2: Train model
    print("\n🤖 Let's train a quick model...")
    user_input = input("Train now? (y/n, default=y): ").strip().lower()

    if user_input != "n":
        trained = run_quick_training()
    else:
        trained = False
        print("⏭️  Skipping training...")

    # Step 3: Test model
    if trained:
        run_quick_test()

    # Step 4: Interactive demo
    print("\n🎮 Want to try the interactive demo?")
    user_input = input("Run demo? (y/n, default=y): ").strip().lower()

    if user_input != "n":
        run_interactive_demo()

    # Step 5: Show next steps
    show_next_steps()

    print("\n🎉 Getting started complete!")
    print("💡 Check out README.md for detailed documentation")
    print("⭐ Don't forget to star the repository if you found it helpful!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for trying BERT Fake News Detection!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Check README.md for troubleshooting tips")
