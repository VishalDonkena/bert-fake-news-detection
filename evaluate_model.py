#!/usr/bin/env python3
"""
BERT Fake News Detection - Model Evaluation & Testing Script

This script provides comprehensive evaluation and testing capabilities for
trained BERT models on fake news detection tasks.

Features:
- Load and evaluate trained models
- Generate detailed performance metrics
- Interactive testing with custom text
- Batch prediction on new datasets
- Model comparison utilities
- Export predictions and reports

Usage:
    python evaluate_model.py --model_path ./models_bert_fast
    python evaluate_model.py --model_path ./models_bert_fast --test_data ./data/test.csv
    python evaluate_model.py --interactive
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class NewsDataset(Dataset):
    """Dataset class for news articles"""

    def __init__(self, texts, labels=None, tokenizer=None, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        result = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[item], dtype=torch.long)

        return result


class ModelEvaluator:
    """Main class for model evaluation and testing"""

    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"🤖 Loading model from: {self.model_path}")

        try:
            # Try loading as BERT first
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        except:
            try:
                # Fallback to AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                )
            except Exception as e:
                raise Exception(f"Failed to load model: {str(e)}")

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully!")
        print(f"💻 Using device: {self.device}")

    def predict_single(self, text, return_probabilities=False):
        """Predict on a single text input"""

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        predicted_class = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
        confidence = torch.max(predictions, dim=-1).values.cpu().numpy()[0]

        result = {
            "prediction": "Fake News" if predicted_class == 1 else "Real News",
            "prediction_id": predicted_class,
            "confidence": confidence,
        }

        if return_probabilities:
            result["probabilities"] = {
                "Real News": predictions[0][0].cpu().numpy(),
                "Fake News": predictions[0][1].cpu().numpy(),
            }

        return result

    def predict_batch(self, texts, batch_size=16, show_progress=True):
        """Predict on a batch of texts"""

        dataset = NewsDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        probabilities = []

        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            if show_progress and batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{total_batches}")

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                batch_probabilities = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                )

                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probabilities.cpu().numpy())

        return predictions, probabilities

    def evaluate_dataset(
        self, df, text_column="text", label_column="label", batch_size=16
    ):
        """Evaluate model on a labeled dataset"""

        print(f"📊 Evaluating on dataset with {len(df)} samples...")

        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()

        predictions, probabilities = self.predict_batch(texts, batch_size)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted"
        )

        # ROC AUC (for binary classification)
        if len(set(true_labels)) == 2:
            prob_positive = [
                p[1] for p in probabilities
            ]  # Probability of class 1 (Fake News)
            auc_score = roc_auc_score(true_labels, prob_positive)
        else:
            auc_score = None

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_score": auc_score,
            "predictions": predictions,
            "probabilities": probabilities,
            "true_labels": true_labels,
        }

        return results

    def generate_report(self, results, save_path=None):
        """Generate comprehensive evaluation report"""

        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)

        print(f"🎯 Overall Performance:")
        print(f"   • Accuracy:  {results['accuracy']:.4f}")
        print(f"   • Precision: {results['precision']:.4f}")
        print(f"   • Recall:    {results['recall']:.4f}")
        print(f"   • F1-Score:  {results['f1_score']:.4f}")

        if results.get("auc_score"):
            print(f"   • ROC AUC:   {results['auc_score']:.4f}")

        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(
            classification_report(
                results["true_labels"],
                results["predictions"],
                target_names=["Real News", "Fake News"],
            )
        )

        # Confusion matrix
        cm = confusion_matrix(results["true_labels"], results["predictions"])
        print(f"\n📈 Confusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    Real    Fake")
        print(f"Real     {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"Fake     {cm[1][0]:5d}   {cm[1][1]:5d}")

        # Save detailed report
        if save_path:
            self.save_detailed_report(results, save_path)
            print(f"\n💾 Detailed report saved to: {save_path}")

        return results

    def save_detailed_report(self, results, save_path):
        """Save detailed evaluation results"""

        # Create results directory
        os.makedirs(save_path, exist_ok=True)

        # Save metrics
        metrics = {
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1_score": float(results["f1_score"]),
            "auc_score": float(results["auc_score"]) if results["auc_score"] else None,
            "timestamp": datetime.now().isoformat(),
        }

        with open(f"{save_path}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions
        predictions_df = pd.DataFrame(
            {
                "true_label": results["true_labels"],
                "predicted_label": results["predictions"],
                "prob_real": [p[0] for p in results["probabilities"]],
                "prob_fake": [p[1] for p in results["probabilities"]],
            }
        )
        predictions_df.to_csv(f"{save_path}/predictions.csv", index=False)

        # Generate and save plots
        self.create_evaluation_plots(results, save_path)

    def create_evaluation_plots(self, results, save_path):
        """Create and save evaluation plots"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(results["true_labels"], results["predictions"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_ylabel("True Label")
        axes[0, 0].set_xlabel("Predicted Label")

        # ROC Curve (if applicable)
        if results.get("auc_score"):
            prob_positive = [p[1] for p in results["probabilities"]]
            fpr, tpr, _ = roc_curve(results["true_labels"], prob_positive)
            axes[0, 1].plot(
                fpr, tpr, "b-", label=f"ROC AUC = {results['auc_score']:.3f}"
            )
            axes[0, 1].plot([0, 1], [0, 1], "r--")
            axes[0, 1].set_xlim([0, 1])
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("True Positive Rate")
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "ROC Curve\nNot Available\n(Multi-class)",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )

        # Prediction Confidence Distribution
        prob_fake = [p[1] for p in results["probabilities"]]
        axes[1, 0].hist(prob_fake, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("Confidence (Probability of Fake News)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Prediction Confidence Distribution")

        # Performance Metrics Bar Chart
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            results["accuracy"],
            results["precision"],
            results["recall"],
            results["f1_score"],
        ]
        bars = axes[1, 1].bar(
            metrics, values, color=["skyblue", "lightgreen", "lightcoral", "gold"]
        )
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_title("Performance Metrics")
        axes[1, 1].set_ylabel("Score")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(f"{save_path}/evaluation_plots.png", dpi=300, bbox_inches="tight")
        plt.close()

    def interactive_testing(self):
        """Interactive testing interface"""

        print("\n🎯 INTERACTIVE TESTING MODE")
        print("=" * 50)
        print("Enter news articles to test (type 'quit' to exit)")
        print("=" * 50)

        while True:
            print("\n📰 Enter your news article:")
            user_input = input("> ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("❌ Please enter some text!")
                continue

            # Get prediction
            result = self.predict_single(user_input, return_probabilities=True)

            # Display results
            print("\n" + "=" * 60)
            print("🔮 PREDICTION RESULTS")
            print("=" * 60)

            # Truncate text for display
            display_text = (
                user_input[:200] + "..." if len(user_input) > 200 else user_input
            )
            print(f"📄 Text: {display_text}")

            print(f"\n🎯 Prediction: **{result['prediction']}**")
            print(f"📊 Confidence: {result['confidence']:.2%}")

            print(f"\n📈 Probabilities:")
            print(f"   • Real News: {result['probabilities']['Real News']:.2%}")
            print(f"   • Fake News: {result['probabilities']['Fake News']:.2%}")

            # Confidence interpretation
            confidence_level = result["confidence"]
            if confidence_level > 0.9:
                interpretation = "Very high confidence"
                emoji = "🎯"
            elif confidence_level > 0.7:
                interpretation = "High confidence"
                emoji = "✅"
            elif confidence_level > 0.6:
                interpretation = "Moderate confidence"
                emoji = "⚠️"
            else:
                interpretation = "Low confidence - uncertain"
                emoji = "❓"

            print(f"\n{emoji} Interpretation: {interpretation}")
            print("=" * 60)


def main():
    """Main function to handle command line arguments"""

    parser = argparse.ArgumentParser(
        description="BERT Fake News Detection - Model Evaluation & Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --model_path ./models_bert_fast
  python evaluate_model.py --model_path ./models_bert_fast --test_data ./data/test.csv
  python evaluate_model.py --model_path ./models_bert_fast --interactive
  python evaluate_model.py --model_path ./models_bert_fast --text "Your news article here"
        """,
    )

    parser.add_argument(
        "--model_path", required=True, help="Path to trained model directory"
    )
    parser.add_argument("--test_data", help="Path to test dataset CSV file")
    parser.add_argument(
        "--text_column",
        default="text",
        help="Name of text column in test data (default: text)",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        help="Name of label column in test data (default: label)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive testing mode"
    )
    parser.add_argument("--text", help="Single text to predict on")

    args = parser.parse_args()

    # Initialize evaluator
    print("🚀 Initializing BERT Model Evaluator...")
    evaluator = ModelEvaluator(args.model_path)

    # Single text prediction
    if args.text:
        print(f"\n📰 Predicting on single text...")
        result = evaluator.predict_single(args.text, return_probabilities=True)

        print("\n" + "=" * 60)
        print("🔮 PREDICTION RESULTS")
        print("=" * 60)
        print(f"📄 Text: {args.text}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"📈 Probabilities:")
        print(f"   • Real News: {result['probabilities']['Real News']:.2%}")
        print(f"   • Fake News: {result['probabilities']['Fake News']:.2%}")
        return

    # Interactive mode
    if args.interactive:
        evaluator.interactive_testing()
        return

    # Dataset evaluation
    if args.test_data:
        if not os.path.exists(args.test_data):
            print(f"❌ Error: Test data file not found: {args.test_data}")
            return

        print(f"📊 Loading test data from: {args.test_data}")
        df = pd.read_csv(args.test_data)

        print(f"✅ Loaded {len(df)} samples")
        print(
            f"📈 Label distribution: {df[args.label_column].value_counts().to_dict()}"
        )

        # Evaluate
        results = evaluator.evaluate_dataset(
            df,
            text_column=args.text_column,
            label_column=args.label_column,
            batch_size=args.batch_size,
        )

        # Generate report
        evaluator.generate_report(results, args.output_dir)

        print(f"\n🎉 Evaluation complete!")
        print(f"📁 Results saved to: {args.output_dir}")
    else:
        print("\n❓ No evaluation task specified!")
        print(
            "Use --test_data for dataset evaluation, --interactive for testing, or --text for single prediction"
        )
        print("Run with --help for more options")


if __name__ == "__main__":
    main()
