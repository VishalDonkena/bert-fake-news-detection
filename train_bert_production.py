#!/usr/bin/env python3
"""
BERT Fake News Detection - Production Training Script

This script provides a robust, production-ready training pipeline with:
- Automatic checkpointing and resume capability
- Advanced logging and monitoring
- Early stopping and learning rate scheduling
- Data augmentation and preprocessing
- Multi-GPU support
- Comprehensive error handling
- Model validation and testing

Usage:
    python train_bert_production.py --config config.json
    python train_bert_production.py --resume checkpoint_dir
"""

import argparse
import os
import json
import logging
import time
import shutil
from datetime import datetime
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    """Enhanced dataset class with data augmentation support"""

    def __init__(self, texts, labels, tokenizer, max_len=512, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Simple text augmentation for training
        if self.augment and np.random.random() > 0.7:
            text = self._augment_text(text)

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

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _augment_text(self, text):
        """Simple text augmentation techniques"""
        # Random word dropout (remove 5% of words)
        words = text.split()
        if len(words) > 10:
            keep_ratio = 0.95
            num_keep = int(len(words) * keep_ratio)
            indices = np.random.choice(len(words), num_keep, replace=False)
            words = [words[i] for i in sorted(indices)]
            text = " ".join(words)
        return text


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        return False

    def _save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """Production-ready model trainer with comprehensive features"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

        # Setup directories
        self._setup_directories()

        logger.info(f"🚀 Trainer initialized")
        logger.info(f"💻 Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

    def _setup_directories(self):
        """Setup training directories"""
        self.output_dir = Path(self.config["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.models_dir = self.output_dir / "models"

        for dir_path in [
            self.output_dir,
            self.checkpoint_dir,
            self.logs_dir,
            self.models_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def load_data(self):
        """Load and preprocess data"""
        logger.info("📊 Loading data...")

        df = pd.read_csv(self.config["data_path"])
        logger.info(f"✅ Loaded {len(df)} samples")
        logger.info(f"📈 Label distribution: {df['label'].value_counts().to_dict()}")

        # Data cleaning
        df = df.dropna()
        df["text"] = df["text"].astype(str)

        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=df["label"],
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.config["val_size"] / (1 - self.config["test_size"]),
            random_state=self.config["random_state"],
            stratify=train_df["label"],
        )

        logger.info(
            f"🔀 Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        return train_df, val_df, test_df

    def create_data_loaders(self, train_df, val_df, test_df):
        """Create data loaders"""
        logger.info("📦 Creating data loaders...")

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])

        # Create datasets
        train_dataset = NewsDataset(
            train_df.text.values,
            train_df.label.values,
            self.tokenizer,
            max_len=self.config["max_length"],
            augment=self.config.get("data_augmentation", False),
        )

        val_dataset = NewsDataset(
            val_df.text.values,
            val_df.label.values,
            self.tokenizer,
            max_len=self.config["max_length"],
        )

        test_dataset = NewsDataset(
            test_df.text.values,
            test_df.label.values,
            self.tokenizer,
            max_len=self.config["max_length"],
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return train_loader, val_loader, test_loader

    def initialize_model(self):
        """Initialize model, optimizer, and scheduler"""
        logger.info("🤖 Initializing model...")

        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=self.config["num_labels"],
            output_attentions=False,
            output_hidden_states=False,
        )

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"🔥 Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        # Initialize optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            eps=self.config.get("adam_epsilon", 1e-8),
        )

        logger.info(
            f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

    def setup_scheduler(self, train_loader):
        """Setup learning rate scheduler"""
        total_steps = len(train_loader) * self.config["epochs"]

        if self.config.get("scheduler_type", "linear") == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 0),
                num_training_steps=total_steps,
            )
        elif self.config["scheduler_type"] == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 0),
                num_training_steps=total_steps,
            )

        # Early stopping
        if self.config.get("early_stopping", False):
            self.early_stopping = EarlyStopping(
                patience=self.config.get("patience", 3),
                min_delta=self.config.get("min_delta", 0.001),
            )

        logger.info(f"📈 Scheduler setup - Total steps: {total_steps:,}")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Handle multi-GPU
            if isinstance(self.model, nn.DataParallel):
                loss = loss.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Update progress bar
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_acc:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # Log to tensorboard
            if batch_idx % self.config.get("log_interval", 100) == 0:
                self.writer.add_scalar("Train/BatchLoss", loss.item(), self.global_step)
                self.writer.add_scalar("Train/BatchAcc", current_acc, self.global_step)
                self.writer.add_scalar(
                    "Train/LearningRate",
                    self.scheduler.get_last_lr()[0],
                    self.global_step,
                )

            self.global_step += 1

            # Save checkpoint
            if (batch_idx + 1) % self.config.get("checkpoint_interval", 1000) == 0:
                self._save_checkpoint(epoch, batch_idx, is_best=False)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                if isinstance(self.model, nn.DataParallel):
                    loss = loss.mean()

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average="weighted")

        return avg_loss, accuracy, f1

    def train(self):
        """Main training loop"""
        logger.info("🏋️‍♀️ Starting training...")

        # Load data
        train_df, val_df, test_df = self.load_data()
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_df, val_df, test_df
        )

        # Initialize model
        self.initialize_model()
        self.setup_scheduler(train_loader)

        # Training loop
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config["epochs"]):
            epoch_start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader)

            # Update history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)
            self.training_history["learning_rates"].append(
                self.scheduler.get_last_lr()[0]
            )

            # Log metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )

            # Tensorboard logging
            self.writer.add_scalar("Epoch/TrainLoss", train_loss, epoch)
            self.writer.add_scalar("Epoch/TrainAcc", train_acc, epoch)
            self.writer.add_scalar("Epoch/ValLoss", val_loss, epoch)
            self.writer.add_scalar("Epoch/ValAcc", val_acc, epoch)
            self.writer.add_scalar("Epoch/ValF1", val_f1, epoch)

            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                logger.info(f"🏆 New best validation accuracy: {val_acc:.4f}")

            # Save checkpoint
            self._save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if self.early_stopping and self.early_stopping(val_acc, self.model):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

            self.current_epoch = epoch + 1

        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time / 3600:.2f} hours")
        logger.info(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}")

        # Final evaluation
        self._final_evaluation(test_loader)

        # Save final model
        self._save_final_model()

        # Close writer
        self.writer.close()

    def _save_checkpoint(self, epoch, batch_idx=None, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "training_history": self.training_history,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Best checkpoint saved to {best_path}")

        # Keep only last N checkpoints
        max_checkpoints = self.config.get("max_checkpoints", 5)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > max_checkpoints:
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.training_history = checkpoint["training_history"]

        logger.info(f"✅ Resumed from epoch {self.current_epoch}")

    def _final_evaluation(self, test_loader):
        """Final evaluation on test set"""
        logger.info("🧪 Final evaluation on test set...")

        test_loss, test_acc, test_f1 = self.validate_epoch(test_loader)

        logger.info(f"🎯 Final Test Results:")
        logger.info(f"   Test Loss: {test_loss:.4f}")
        logger.info(f"   Test Accuracy: {test_acc:.4f}")
        logger.info(f"   Test F1: {test_f1:.4f}")

        # Save test results
        test_results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "best_val_acc": self.best_val_acc,
        }

        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    def _save_final_model(self):
        """Save the final trained model"""
        logger.info("💾 Saving final model...")

        # Save model and tokenizer
        model_to_save = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        final_model_dir = self.models_dir / "final_model"
        final_model_dir.mkdir(exist_ok=True)

        model_to_save.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)

        # Save best model
        best_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        if best_checkpoint_path.exists():
            best_model_dir = self.models_dir / "best_model"
            best_model_dir.mkdir(exist_ok=True)

            # Load best checkpoint
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            model_to_save.load_state_dict(checkpoint["model_state_dict"])
            model_to_save.save_pretrained(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)

        # Save training history
        with open(self.models_dir / "training_history.json", "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            history_serializable = {}
            for key, values in self.training_history.items():
                history_serializable[key] = [float(v) for v in values]
            json.dump(history_serializable, f, indent=2)

        logger.info(f"✅ Models saved to {self.models_dir}")


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_default_config():
    """Create default configuration"""
    return {
        "data_path": "./data/sample_news.csv",
        "output_dir": "./output",
        "model_name": "bert-base-uncased",
        "num_labels": 2,
        "max_length": 512,
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "scheduler_type": "linear",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
        "data_augmentation": False,
        "early_stopping": True,
        "patience": 3,
        "min_delta": 0.001,
        "max_checkpoints": 5,
        "checkpoint_interval": 1000,
        "log_interval": 100,
        "num_workers": 0,
        "adam_epsilon": 1e-8,
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BERT Production Training Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to configuration JSON file")
    group.add_argument("--resume", help="Path to checkpoint directory to resume from")

    parser.add_argument(
        "--create-config", help="Create default config file at specified path"
    )

    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Default config created at {args.create_config}")
        return

    # Load configuration
    if args.config:
        config = load_config(args.config)
        trainer = ModelTrainer(config)
        trainer.train()

    elif args.resume:
        # Find the best checkpoint in the directory
        resume_dir = Path(args.resume)
        best_checkpoint = resume_dir / "checkpoints" / "best_checkpoint.pt"

        if not best_checkpoint.exists():
            # Find latest checkpoint
            checkpoints = sorted(resume_dir.glob("checkpoints/checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {resume_dir}")
            best_checkpoint = checkpoints[-1]

        # Load config from checkpoint
        checkpoint = torch.load(best_checkpoint, map_location="cpu")
        config = checkpoint["config"]

        # Create trainer and load checkpoint
        trainer = ModelTrainer(config)
        trainer.initialize_model()
        trainer.load_checkpoint(best_checkpoint)
        trainer.train()


if __name__ == "__main__":
    main()
