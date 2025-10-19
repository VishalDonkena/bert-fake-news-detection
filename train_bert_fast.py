import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# --- Configuration for FAST training ---
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128  # Reduced from 256 for faster processing
BATCH_SIZE = 8  # Reduced batch size for faster processing
EPOCHS = 1  # Just 1 epoch for quick testing
LEARNING_RATE = 5e-5  # Higher learning rate for faster convergence
MODEL_SAVE_PATH = "./models_bert_fast"
DATA_PATH = "./data/sample_news.csv"

# Limit dataset size for faster training
MAX_SAMPLES = 1000  # Only use first 1000 samples

# --- 1. Load and Prepare Dataset ---


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

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
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NewsDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_samples = 0

    for batch_idx, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return correct_predictions.double() / total_samples, np.mean(losses)


def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            losses.append(loss.item())

    return correct_predictions.double() / total_samples, np.mean(losses)


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Limit dataset size for faster training
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"Using only {MAX_SAMPLES} samples for faster training")

    print(f"Dataset size: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts()}")

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(df_train)}, Validation samples: {len(df_val)}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # --- 2. Initialize Model, Optimizer, and Scheduler ---

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # --- 3. Training Loop ---
    print("Starting fast training...")
    print(
        f"Configuration: MAX_LENGTH={MAX_LENGTH}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}"
    )

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, device, scheduler
        )
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, val_data_loader, device)
        print(f"Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # --- 4. Save the Model ---

    print("\nSaving model...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("Fast training complete!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Final validation accuracy: {val_acc:.4f}")
