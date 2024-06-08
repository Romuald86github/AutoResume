import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Input, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def train_cosine_similarity_model(X_resumes, X_jd, vectorizer_resumes, vectorizer_jd):
    # Transform the raw text data using the vectorizers
    X_resumes_transformed = vectorizer_resumes.transform(X_resumes)
    X_jd_transformed = vectorizer_jd.transform(X_jd)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(X_resumes_transformed.toarray(), X_jd_transformed.toarray())
    joblib.dump(similarity_matrix, 'models/cosine_similarity_model.pkl')

def train_semantic_similarity_model(X_resumes, X_jd, max_words=5000, max_len=500):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_resumes + X_jd)

    X_resumes_seq = tokenizer.texts_to_sequences(X_resumes)
    X_jd_seq = tokenizer.texts_to_sequences(X_jd)

    X_resumes_pad = pad_sequences(X_resumes_seq, maxlen=max_len)
    X_jd_pad = pad_sequences(X_jd_seq, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit([X_resumes_pad, X_jd_pad], np.ones(len(X_resumes_pad)), epochs=5, batch_size=64, validation_split=0.2)
    model.save('models/semantic_similarity_model.h5')

def train_siamese_model(X_resumes, X_jd, max_words=5000, max_len=500):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_resumes + X_jd)

    X_resumes_seq = tokenizer.texts_to_sequences(X_resumes)
    X_jd_seq = tokenizer.texts_to_sequences(X_jd)

    X_resumes_pad = pad_sequences(X_resumes_seq, maxlen=max_len)
    X_jd_pad = pad_sequences(X_jd_seq, maxlen=max_len)

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    resume_input = Input(shape=(max_len,))
    jd_input = Input(shape=(max_len,))

    shared_lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2)
    resume_embedding = shared_lstm(resume_input)
    jd_embedding = shared_lstm(jd_input)

    distance = Lambda(euclidean_distance)([resume_embedding, jd_embedding])
    model = Model(inputs=[resume_input, jd_input], outputs=distance)
    model.compile(loss=MeanSquaredError(), optimizer=Adam())
    model.fit([X_resumes_pad, X_jd_pad], np.zeros(len(X_resumes_pad)), epochs=5, batch_size=64, validation_split=0.2)
    model.save('models/siamese_model.h5')

def train_ranking_model(X_resumes, X_jd, max_words=5000, max_len=500):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_resumes + X_jd)

    X_resumes_seq = tokenizer.texts_to_sequences(X_resumes)
    X_jd_seq = tokenizer.texts_to_sequences(X_jd)

    X_resumes_pad = pad_sequences(X_resumes_seq, maxlen=max_len)
    X_jd_pad = pad_sequences(X_jd_seq, maxlen=max_len)

    # Define the ranking model architecture
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    # Train the ranking model
    model.fit([X_resumes_pad, X_jd_pad], np.arange(len(X_resumes_pad)), epochs=5, batch_size=64, validation_split=0.2)
    model.save('models/ranking_model.h5')

if __name__ == "__main__":
    resumes_data, jd_data = joblib.load('data/resume_vectors.pkl')
    jd_data, _ = joblib.load('data/job_description_vectors.pkl')

    X_resumes = resumes_data['raw_texts']
    X_jd = jd_data['raw_texts']

    vectorizer_resumes = resumes_data['vectorizer']
    vectorizer_jd = jd_data['vectorizer']

    train_cosine_similarity_model(X_resumes, X_jd, vectorizer_resumes, vectorizer_jd)
    train_semantic_similarity_model(X_resumes, X_jd)
    train_siamese_model(X_resumes, X_jd)
    train_ranking_model(X_resumes, X_jd)
