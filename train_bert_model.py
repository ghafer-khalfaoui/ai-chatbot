import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import pickle
import os # To manage saving paths

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased" # Smaller, faster BERT variant
MAX_LEN = 32 # Max sequence length for BERT (shorter is often better)
OUTPUT_DIR = "bert_intent_model" # Directory to save the fine-tuned model

# --- 1. Load and Prepare Data ---
print("Loading intents...")
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
except Exception as e:
    print(f"Error loading intents.json: {e}")
    exit()

patterns = []
tags = []
for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(tag)

if not patterns or not tags:
    print("Error: No patterns or tags found in intents.json!")
    exit()

# Encode tags (labels)
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)
num_classes = len(label_encoder.classes_)
print(f"Loaded {len(patterns)} patterns for {num_classes} classes.")

# --- 2. Prepare Data for Hugging Face `datasets` ---
# Create a dictionary suitable for the Dataset object
data_dict = {'text': patterns, 'label': encoded_tags}
dataset = Dataset.from_dict(data_dict)

# --- 3. Load BERT Tokenizer and Tokenize Data ---
print(f"Loading tokenizer for '{MODEL_NAME}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

def tokenize_function(examples):
    # Tokenize the text using padding and truncation
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LEN)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert labels to TensorFlow compatible format if needed (already integers)
# No need to one-hot encode with TFAutoModelForSequenceClassification and sparse_categorical_crossentropy

# --- 4. Load Pre-trained BERT Model for Sequence Classification ---
print(f"Loading pre-trained model '{MODEL_NAME}' for fine-tuning...")
try:
    # Load the model, specifying the number of output labels
   model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes, use_safetensors=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 5. Prepare TensorFlow Dataset ---
print("Preparing TensorFlow dataset...")
# Select necessary columns and convert to TensorFlow format
tf_dataset = tokenized_dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids'], # Features BERT uses
    label_cols=['label'], # The labels
    shuffle=True,
    batch_size=16 # Adjust batch size based on memory
)

# --- 6. Compile and Fine-Tune the Model ---
print("Compiling model...")
# Use Adam optimizer with a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # 5e-5 is a common starting point for BERT fine-tuning
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # Use from_logits=True for TF models
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

print("Fine-tuning the model...")
epochs = 5 # Fewer epochs are often needed for fine-tuning BERT (e.g., 3-5)
history = model.fit(tf_dataset, epochs=epochs, verbose=1)

print("Model fine-tuning complete.")
try:
    accuracy = history.history['accuracy'][-1]
    print(f"Final training accuracy: {accuracy*100:.2f}%")
except KeyError:
     print("Could not retrieve accuracy from history. Check TensorFlow version/output.")


# --- 7. Save the Fine-Tuned Model, Tokenizer, and Label Encoder ---
print(f"Saving fine-tuned model and supporting files to '{OUTPUT_DIR}'...")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the fine-tuned model and tokenizer using Hugging Face's method
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save the label encoder separately using pickle
label_encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pickle')
with open(label_encoder_path, 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save max_len used for tokenizer
max_len_path = os.path.join(OUTPUT_DIR, 'max_len.json')
with open(max_len_path, 'w') as f:
     json.dump({'max_len': MAX_LEN}, f)


print(f"✅ Files saved successfully in '{OUTPUT_DIR}' directory.")