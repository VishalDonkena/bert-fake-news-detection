import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import os

# --- Minimal Configuration for super fast testing ---
MODEL_NAME = "distilbert-base-uncased"  # Smaller, faster model
MAX_LENGTH = 64  # Very short sequences
BATCH_SIZE = 4  # Small batch size
EPOCHS = 1  # Just 1 epoch
LEARNING_RATE = 3e-4  # Higher learning rate
MODEL_SAVE_PATH = "./models_bert_minimal"
DATA_PATH = "./data/sample_news.csv"

# Ultra small dataset for testing
MAX_SAMPLES = 100  # Only 100 samples for super fast training


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])[:200]  # Truncate text early
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
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
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_model(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader), correct / total


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), correct / total


if __name__ == "__main__":
    print("🚀 Starting MINIMAL BERT training for quick testing...")

    # Load and prepare data
    print("📁 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Use only a tiny subset
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42)
    print(f"📊 Using {len(df)} samples")
    print(f"📈 Label distribution:\n{df['label'].value_counts()}")

    # Split data
    df_train, df_val = train_test_split(df, test_size=0.3, random_state=42)
    print(f"🔀 Train: {len(df_train)}, Validation: {len(df_val)}")

    # Initialize tokenizer and model
    print("🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    model.to(device)

    # Create data loaders
    train_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training
    print(f"🏋️‍♀️ Starting training with {EPOCHS} epoch(s)...")
    print(f"⚙️  Config: max_len={MAX_LENGTH}, batch_size={BATCH_SIZE}")

    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, device, scheduler
        )
        print(f"🎯 Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Evaluate
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        print(f"✅ Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save model
    print(f"\n💾 Saving model to {MODEL_SAVE_PATH}...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("🎉 Minimal training complete!")
    print(f"📊 Final validation accuracy: {val_acc:.4f}")
    print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
