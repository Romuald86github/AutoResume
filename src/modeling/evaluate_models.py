import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import ndcg_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError

# Define the LSTMWrapper class
class LSTMWrapper(Layer):
    def __init__(self, units, dropout=0.2, recurrent_dropout=0.2, **kwargs):
        super(LSTMWrapper, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.lstm = tf.keras.layers.LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=False)

    def call(self, inputs):
        return self.lstm(tf.expand_dims(inputs, axis=1))

# Register the euclidean_distance function
@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# Define the EuclideanDistanceLayer
class EuclideanDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return euclidean_distance(inputs)

# Register the mse function
@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Preprocess and tokenize the text data
def preprocess_texts(texts, tokenizer, max_len=500):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences

def evaluate_cosine_similarity_model(X_resumes, X_jd, vectorizer):
    X_resumes_transformed = vectorizer.transform(X_resumes)
    X_jd_transformed = vectorizer.transform(X_jd)

    # Align the feature dimensions
    min_features = min(X_resumes_transformed.shape[1], X_jd_transformed.shape[1])
    X_resumes_transformed = X_resumes_transformed[:, :min_features]
    X_jd_transformed = X_jd_transformed[:, :min_features]

    similarity_matrix = cosine_similarity(X_resumes_transformed.toarray(), X_jd_transformed.toarray())
    mean_cosine_similarity = np.mean(similarity_matrix.diagonal())
    
    print(f"Cosine Similarity Model:")
    print(f"Mean Cosine Similarity: {mean_cosine_similarity:.4f}")
    
    return mean_cosine_similarity

def evaluate_semantic_similarity_model(X_resumes, X_jd):
    model = load_model('models/semantic_similarity_model.h5')
    # Ensure both inputs have the same number of samples
    min_samples = min(len(X_resumes), len(X_jd))
    X_resumes = X_resumes[:min_samples]
    X_jd = X_jd[:min_samples]
    predictions = model.predict([X_resumes, X_jd])
    mean_cosine_similarity = np.mean(predictions)
    
    print(f"Semantic Similarity Model:")
    print(f"Mean Semantic Similarity: {mean_cosine_similarity:.4f}")
    
    return mean_cosine_similarity

def evaluate_siamese_model(X_resumes, X_jd):
    # Use custom_objects to register the custom layer
    model = load_model('models/siamese_model.h5', custom_objects={'LSTMWrapper': LSTMWrapper, 'EuclideanDistanceLayer': EuclideanDistanceLayer})
    # Ensure both inputs have the same number of samples
    min_samples = min(len(X_resumes), len(X_jd))
    X_resumes = X_resumes[:min_samples]
    X_jd = X_jd[:min_samples]
    resume_embeddings = model.predict([X_resumes, X_jd])
    jd_embeddings = model.predict([X_jd, X_resumes])
    mean_euclidean_distance = np.mean(np.sqrt(np.sum((resume_embeddings - jd_embeddings) ** 2, axis=1)))
    
    print(f"Siamese Model:")
    print(f"Mean Euclidean Distance: {mean_euclidean_distance:.4f}")
    
    return -mean_euclidean_distance

def evaluate_ranking_model(X_resumes, X_jd):
    model = load_model('models/ranking_model.h5', custom_objects={'mse': mse})
    y_pred = model.predict(X_resumes).flatten()
    ndcg = ndcg_score([np.arange(len(X_resumes))], [y_pred])
    
    print(f"Ranking Model:")
    print(f"NDCG: {ndcg:.4f}")
    
    return ndcg

if __name__ == "__main__":
    resumes_data = joblib.load('data/resume_vectors.pkl')
    jd_data = joblib.load('data/job_description_vectors.pkl')

    X_resumes = resumes_data['raw_texts']
    X_jd = jd_data['raw_texts']

    vectorizer = resumes_data['vectorizer']

    # Use the same tokenizer for semantic and ranking models to ensure consistency
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_resumes + X_jd)

    # Preprocess the text data
    X_resumes_preprocessed = preprocess_texts(X_resumes, tokenizer)
    X_jd_preprocessed = preprocess_texts(X_jd, tokenizer)

    model_scores = {
        'cosine_similarity_model.pkl': evaluate_cosine_similarity_model(X_resumes, X_jd, vectorizer),
        'semantic_similarity_model.h5': evaluate_semantic_similarity_model(X_resumes_preprocessed, X_jd_preprocessed),
        'siamese_model.h5': evaluate_siamese_model(X_resumes_preprocessed, X_jd_preprocessed),
        'ranking_model.h5': evaluate_ranking_model(X_resumes_preprocessed, X_jd_preprocessed)
    }

    # Select the best model based on the evaluation metrics
    best_model_name = max(model_scores, key=model_scores.get)
    
    # Create the 'models/' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    best_model_path = os.path.join('models', 'best_model')
    if best_model_name.endswith('.pkl'):
        best_model = joblib.load(f'models/{best_model_name}')
        joblib.dump(best_model, f'{best_model_path}.pkl')
    else:
        best_model = load_model(f'models/{best_model_name}', custom_objects={'LSTMWrapper': LSTMWrapper, 'EuclideanDistanceLayer': EuclideanDistanceLayer, 'mse': mse})
        best_model.save(f'{best_model_path}.h5')

    print(f"The best model is {best_model_name} and has been saved as {best_model_path}.")