import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def score_resumes(job_description, resume_dir, vectorizer_file, best_model_file):
    X_jd, _, vectorizer = joblib.load(vectorizer_file)
    jd_vector = vectorizer.transform([job_description])

    X_resumes, filenames, _ = joblib.load(resume_dir)
    if best_model_file.endswith('.pkl'):
        model = joblib.load(best_model_file)
        jd_scores = cosine_similarity(jd_vector, X_resumes)
    else:
        model = load_model(best_model_file)
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(filenames)
        jd_sequences = tokenizer.texts_to_sequences([job_description])
        jd_padded = pad_sequences(jd_sequences, maxlen=500)
        jd_scores = model.predict(jd_padded)
    
    scores = list(zip(filenames, jd_scores.flatten()))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

if __name__ == "__main__":
    job_description = 'Your job description text here'
    resume_dir = '../../data/processed/resume_vectors.pkl'
    vectorizer_file = '../../data/processed/jd_vectors.pkl'
    best_model_file = 'models/best_model.pkl'

    scores = score_resumes(job_description, resume_dir, vectorizer_file, best_model_file)
    for filename, score in scores:
        print(f'{filename}: {score}')