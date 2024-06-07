import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
def train_sklearn_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(),
        'svm': SVC(),
        'random_forest': RandomForestClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{name}.pkl')

def train_lstm_model(X_train, y_train, max_words=5000, max_len=500):
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
    model.save('models/lstm_model.h5')

if __name__ == "__main__":
    X, filenames, vectorizer = joblib.load('../../data/processed/resume_vectors.pkl')
    y = [1] * (len(filenames) // 2) + [0] * (len(filenames) // 2)  # Dummy labels

    train_sklearn_models(X, y)
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(filenames)
    X_lstm = tokenizer.texts_to_sequences(filenames)
    X_lstm = pad_sequences(X_lstm, maxlen=500)
    
    train_lstm_model(X_lstm, y)