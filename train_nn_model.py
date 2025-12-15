import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
import pickle # To save tokenizer and label encoder

# --- 1. Load and Prepare Data ---
print("Loading intents...")
try:
    # Ensure correct encoding for emojis/special characters
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: intents.json not found!")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode intents.json. Check its format.")
    exit()

patterns = []
tags = []
responses_dict = {} # Store responses for later use if needed

for intent in intents['intents']:
    tag = intent['tag']
    # Store responses associated with the tag
    if tag not in responses_dict:
        responses_dict[tag] = intent['responses']
        
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(tag)

if not patterns or not tags:
    print("Error: No patterns or tags found in intents.json!")
    exit()

num_classes = len(set(tags))
print(f"Loaded {len(patterns)} patterns belonging to {num_classes} classes.")

# --- 2. Preprocess Text Data ---
print("Preprocessing text data...")

# Tokenize words (convert words to numbers)
# num_words=2000 means we only keep the top 2000 most frequent words
# oov_token handles words not seen during training
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)

# Pad sequences so they all have the same length
# maxlen defines the maximum length of a sequence
# padding='post' adds padding at the end
# truncating='post' removes elements from the end if sequence is too long
max_len = 20 # Max words per pattern to consider
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Encode tags (convert tag strings to numbers)
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)
encoded_tags = np.array(encoded_tags) # Convert to numpy array

# --- 3. Define the Neural Network Model ---
print("Building the neural network model...")

embedding_dim = 16 # Dimension of the word embedding vectors
vocab_size = len(word_index) + 1 # +1 for the <OOV> token

model = Sequential([
    # Embedding layer: Turns word indices into dense vectors of embedding_dim size
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    
    # GlobalAveragePooling1D: Averages the embeddings for all words in a sequence
    # This is a simple way to handle variable length input for a feedforward network
    GlobalAveragePooling1D(),
    
    # Dense hidden layer with ReLU activation
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout helps prevent overfitting
    
    # Output layer: num_classes neurons, softmax activation for multi-class probability
    Dense(num_classes, activation='softmax')
])

# Alternative Model using LSTM (often better for sequence data like text)
# model = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
#     LSTM(64, return_sequences=True), # Return sequence for next LSTM or pooling
#     LSTM(32),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

model.compile(loss='sparse_categorical_crossentropy', # Good loss for integer labels
              optimizer='adam',
              metrics=['accuracy'])

model.summary() # Print a summary of the model layers

# --- 4. Train the Model ---
print("Training the model...")
epochs = 100 # Number of times to iterate over the entire dataset
# Increase epochs if accuracy is low, decrease if overfitting
history = model.fit(padded_sequences, encoded_tags, epochs=epochs, verbose=1)

print("Model training complete.")
accuracy = history.history['accuracy'][-1]
print(f"Final training accuracy: {accuracy*100:.2f}%")

# --- 5. Save the Model and Supporting Files ---
print("Saving model and supporting files...")

# Save the trained Keras model
model.save('chatbot_nn_model.keras') # Use the recommended .keras format

# Save the tokenizer (needed to process new user input)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder (needed to convert predictions back to tags)
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save max_len, needed for padding new input
with open('max_len.json', 'w') as f:
     json.dump({'max_len': max_len}, f)

print("✅ Files saved successfully: chatbot_nn_model.keras, tokenizer.pickle, label_encoder.pickle, max_len.json")