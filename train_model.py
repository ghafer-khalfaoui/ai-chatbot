import json
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import pickle
import os

# 1. Setup
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 50
os.makedirs("bert_intent_model", exist_ok=True)

# 2. Load Data
with open('intents.json', 'r') as f:
    data = json.load(f)

patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# 3. Encode Labels
encoder = LabelEncoder()
labels = encoder.fit_transform(tags)
num_classes = len(set(labels))

with open('bert_intent_model/label_encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# 4. Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained("bert_intent_model")

encodings = tokenizer(patterns, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='tf')
dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels)).shuffle(100).batch(8)

# 5. Train
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("Training Model...")
model.fit(dataset, epochs=5)

# 6. Save
model.save_pretrained("bert_intent_model")
with open('bert_intent_model/max_len.json', 'w') as f:
    json.dump({"max_len": MAX_LEN}, f)

print("✅ Model trained and saved to 'bert_intent_model/' folder.")