import os
import json
import re
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ===================================================================
# 1. SETTINGS
# ===================================================================

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 10
MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# 2. UTILS
# ===================================================================

def clean_text(text):
    """Clean text: lowercase and remove special chars"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# ===================================================================
# 3. LOAD DATA
# ===================================================================

print("⏳ Loading and processing data...")

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(clean_text(pattern))
        labels.append(intent["tag"])

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

print(f"✅ Found {len(texts)} patterns and {num_classes} unique tags.")

# ===================================================================
# 4. TOKENIZATION
# ===================================================================

print("⏳ Tokenizing data...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

dataset = IntentDataset(texts, y, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================================================================
# 5. MODEL
# ===================================================================

print(f"⏳ Downloading {MODEL_NAME} model...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes
)
model.to(DEVICE)

# ===================================================================
# 6. TRAINING
# ===================================================================

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

print("🚀 Starting training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)

    print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(dataloader):.4f} | Accuracy: {correct/total:.4f}")

# ===================================================================
# 7. SAVE ARTIFACTS
# ===================================================================

print("⏳ Saving artifacts...")

# Save model
model.save_pretrained("bert_intent_model")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Training complete. Files saved:")
print("   - bert_intent_model/ (Model weights)")
print("   - tokenizer.pkl (Tokenizer)")
print("   - label_encoder.pkl (Label mapping)")
