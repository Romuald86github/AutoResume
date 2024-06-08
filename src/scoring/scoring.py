import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import numpy as np
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

def score_resumes(job_description, resume_vectors_file, jd_vectors_file, best_model_path):
    resumes_data = joblib.load(resume_vectors_file)
    jd_data = joblib.load(jd_vectors_file)

    X_resumes = resumes_data['raw_texts']
    X_jd = jd_data['raw_texts']
    vectorizer = resumes_data['vectorizer']

    # Use the same tokenizer for semantic and ranking models to ensure consistency
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_resumes + X_jd)

    # Preprocess the job description
    preprocessed_jd = preprocess_texts([job_description], tokenizer)

    # Load the best model
    if best_model_path.endswith('.pkl'):
        # Load the cosine similarity or ranking model
        best_model = joblib.load(best_model_path)
        if best_model_path == 'models/best_model.pkl':
            X_resumes_transformed = vectorizer.transform(X_resumes)
            X_jd_transformed = vectorizer.transform([job_description])
            similarity_matrix = best_model.toarray()
            scores = similarity_matrix[:, 0]
        else:
            scores = best_model.predict([X_resumes, preprocessed_jd]).flatten()
    else:
        # Load the semantic similarity or siamese model
        custom_objects = {'LSTMWrapper': LSTMWrapper, 'EuclideanDistanceLayer': EuclideanDistanceLayer, 'mse': mse}
        best_model = load_model(best_model_path, custom_objects=custom_objects)
        if best_model_path == 'models/best_model.h5':
            X_resumes_preprocessed = preprocess_texts(X_resumes, tokenizer)
            scores = best_model.predict([X_resumes_preprocessed, preprocessed_jd])
        else:
            X_resumes_preprocessed = preprocess_texts(X_resumes, tokenizer)
            resume_embeddings = best_model.predict([X_resumes_preprocessed, preprocessed_jd])
            jd_embeddings = best_model.predict([preprocessed_jd, X_resumes_preprocessed])
            scores = -np.mean(np.sqrt(np.sum((resume_embeddings - jd_embeddings) ** 2, axis=1)), axis=0)

    # Sort the resumes based on the scores
    sorted_indices = np.argsort(scores)[::-1]
    sorted_filenames = [os.path.basename(x) for x in np.array(X_resumes)[sorted_indices]]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Return the sorted resumes and their scores
    return dict(zip(sorted_filenames, sorted_scores))

if __name__ == "__main__":
    job_description = "Software Engineer with experience in Python and machine learning."
    scores = score_resumes(job_description, 'data/resume_vectors.pkl', 'data/job_description_vectors.pkl', 'models/best_model.h5')

    for filename, score in scores.items():
        print(f"{filename}: {score:.4f}")