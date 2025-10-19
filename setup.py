#!/usr/bin/env python3
"""
BERT Fake News Detection - Setup Script

This script helps users quickly set up the project environment
and verify everything is working correctly.

Usage:
    python setup.py
    python setup.py --check
    python setup.py --demo
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def print_header():
    """Print welcome header"""
    print("🚀 BERT Fake News Detection - Setup")
    print("=" * 50)


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")

    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher required")
        print(f"   Current version: {sys.version}")
        return False

    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to install requirements")
        return False


def check_packages():
    """Check if required packages are installed"""
    print("\n🔍 Checking package installation...")

    required_packages = [
        "torch",
        "transformers",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False

    print("✅ All packages installed")
    return True


def check_gpu():
    """Check GPU availability"""
    print("\n🔥 Checking GPU availability...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name}")
            print(f"   Memory: {memory:.1f} GB")
            return True
        else:
            print("⚠️  No GPU available - will use CPU")
            print("   Consider using Google Colab for GPU training")
            return False
    except ImportError:
        print("❌ Error: Cannot check GPU (torch not installed)")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")

    dirs_to_create = ["output", "logs", "checkpoints"]

    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_name}/")
        else:
            print(f"✅ Exists: {dir_name}/")


def check_data():
    """Check if sample data exists"""
    print("\n📊 Checking data files...")

    data_file = Path("data/sample_news.csv")
    if data_file.exists():
        print("✅ Sample data found: data/sample_news.csv")

        # Check data format
        try:
            import pandas as pd

            df = pd.read_csv(data_file)
            print(f"   Samples: {len(df)}")
            print(f"   Columns: {df.columns.tolist()}")

            if "text" in df.columns and "label" in df.columns:
                print("✅ Data format is correct")
                return True
            else:
                print("⚠️  Data format issue - missing 'text' or 'label' columns")
                return False
        except Exception as e:
            print(f"⚠️  Could not read data file: {e}")
            return False
    else:
        print("⚠️  Sample data not found")
        print("   You'll need to provide your own news.csv file")
        return False


def run_demo():
    """Run a quick demo to verify everything works"""
    print("\n🧪 Running quick demo...")

    # Check if we have a trained model
    model_paths = [
        Path("models_bert_minimal"),
        Path("models_bert_fast"),
        Path("output/models/best_model"),
    ]

    model_found = None
    for path in model_paths:
        if path.exists():
            model_found = path
            break

    if not model_found:
        print("⚠️  No trained model found")
        print("   Run training first: python train_bert_minimal.py")
        return False

    # Try to run a simple test
    try:
        print(f"🎯 Testing model: {model_found}")

        # Import and test
        from evaluate_model import ModelEvaluator

        evaluator = ModelEvaluator(str(model_found))
        result = evaluator.predict_single(
            "This is a test news article about technology.", return_probabilities=True
        )

        print(f"✅ Demo successful!")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("\n🎯 Next Steps:")
    print("=" * 30)
    print("1. Train a model:")
    print("   python train_bert_minimal.py")
    print("")
    print("2. Test the model:")
    print("   python test_model.py --quick")
    print("")
    print("3. Interactive testing:")
    print(
        "   python evaluate_model.py --model_path ./models_bert_minimal --interactive"
    )
    print("")
    print("4. Google Colab training (for better performance):")
    print("   Upload BERT_Training_Colab.ipynb to Google Colab")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="BERT Fake News Detection Setup")
    parser.add_argument("--check", action="store_true", help="Only check installation")
    parser.add_argument("--demo", action="store_true", help="Run demo after setup")

    args = parser.parse_args()

    print_header()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements if not just checking
    if not args.check:
        if not install_requirements():
            sys.exit(1)

    # Check packages
    if not check_packages():
        if args.check:
            print("\n💡 Run without --check to install requirements")
        sys.exit(1)

    # Check GPU
    check_gpu()

    # Create directories
    if not args.check:
        create_directories()

    # Check data
    check_data()

    # Run demo if requested
    if args.demo:
        run_demo()

    print("\n🎉 Setup complete!")

    if not args.check and not args.demo:
        print_next_steps()


if __name__ == "__main__":
    main()
