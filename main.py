# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample Data 
data = pd.DataFrame({
    'text': [
        'This is a true news article about an event.',
        'This is a fake news article that misleads people.',
    ],
    'label': [0, 1]
})

# Preprocessing the text data
def preprocess_text(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(preprocess_text)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad sequences to ensure uniform input length
max_length = 100  # will adjust this based on average length of articles
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to NumPy array
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluating the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# testing
def predict_fake_news(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    prediction = model.predict(padded_sequence)
    return "Fake" if prediction > 0.5 else "Real"

print(predict_fake_news("Some fake news about a misleading event"))