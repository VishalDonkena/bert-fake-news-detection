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

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = (
    256  # BERT has a limit of 512, but 256 is often sufficient and more efficient
)
BATCH_SIZE = 16  # Adjust based on your GPU memory
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = "/Users/vishaldonkena/Code/fake_news_detector_bert/models_bert"
DATA_PATH = "/Users/vishaldonkena/Code/fake_news_detector/data/processed/news.csv"

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
    return DataLoader(
        ds, batch_size=batch_size, num_workers=0
    )  # Changed num_workers to 0


def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0

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
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

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
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# --- 2. Prediction Function ---

def predict(text, model, tokenizer, device, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        prediction = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

    label = "Fake News" if prediction == 1 else "Real News"
    confidence = probs[prediction]

    return label, confidence


if __name__ == "__main__":
    # --- Training Phase ---
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, device, scheduler
        )
        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_model(model, val_data_loader, device)
        print(f"Val   loss {val_loss} accuracy {val_acc}")
        print()

    print("Saving model...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Training complete.")

    # --- Prediction Phase ---
    print("=" * 60)
    print("🔍 FAKE NEWS DETECTOR (BERT Edition)")
    print("=" * 60)
    print("Loading fine-tuned BERT model for prediction...")

    # No need to load the model again, it's already in memory.
    # tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    # model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    # model = model.to(device)

    model.eval()
    print("Model loaded and ready for prediction.")

    print("Enter a news article or a short statement to analyze.")

    while True:
        print("
Enter text (or type 'exit' to quit):")
        try:
            article = input("> ")
            if article.lower() == "exit":
                break
            if not article.strip():
                continue

            label, confidence = predict(article, model, tokenizer, device, MAX_LENGTH)
            print(f"
📰 Prediction: {label} (Confidence: {confidence:.2%})")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("
Thank you for using the Fake News Detector!")
