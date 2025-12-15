import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib # Used to save our model

# --- Download NLTK data ---
try:
    # Check for 'wordnet' (for lemmatization)
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK 'wordnet' not found. Downloading...")
    nltk.download('wordnet')

try:
    # Check for 'punkt' (for tokenization)
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')

try:
    # Check for 'punkt_tab' (the missing resource)
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab')

# --- 1. Load and Preprocess Data ---
print("Loading intents...")
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# These lists will hold our training data
patterns = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize each word in the pattern
        words = nltk.word_tokenize(pattern)
        lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in words]

        # Add the processed pattern and its tag
        patterns.append(" ".join(lemmatized_words))
        tags.append(intent['tag'])

print(f"Loaded {len(patterns)} patterns.")

# --- 2. Create and Train the Model Pipeline ---
print("Creating and training the model...")

# This pipeline does two things:
# 1. TfidfVectorizer: Converts our text patterns into a matrix of numbers.
# 2. MultinomialNB: A simple but effective classifier for text.
model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model on our patterns and tags
model_pipeline.fit(patterns, tags)

print("Model training complete.")

# --- 3. Save the Trained Model ---
model_filename = 'chatbot_model.joblib'
joblib.dump(model_pipeline, model_filename)

print(f"Model saved successfully as '{model_filename}'")