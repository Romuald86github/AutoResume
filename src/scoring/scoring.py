import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

def score_resumes(job_description, resume_vectors_file, jd_vectors_file, best_model_file):
    _, X_resumes, vectorizer_resumes = joblib.load(resume_vectors_file)
    _, X_jd, vectorizer_jd = joblib.load(jd_vectors_file)

    # Preprocess the job description
    preprocessed_jd = vectorizer_jd.transform([job_description])

    # Load the best model
    if best_model_file.endswith('.pkl'):
        # Load the cosine similarity or ranking model
        best_model = joblib.load(best_model_file)
        if best_model_file == 'models/cosine_similarity_model.pkl':
            scores = best_model.diagonal()
        else:
            scores = best_model.predict([X_resumes, preprocessed_jd.toarray()]).flatten()
    else:
        # Load the semantic similarity or siamese model
        best_model = load_model(best_model_file)
        scores = best_model.predict([X_resumes, preprocessed_jd.toarray()])

    # Sort the resumes based on the scores
    sorted_indices = np.argsort(scores)[::-1]
    sorted_filenames = [os.path.basename(f) for f in X_resumes[sorted_indices]]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Return the sorted resumes and their scores
    return dict(zip(sorted_filenames, sorted_scores))

if __name__ == "__main__":
    job_description = "Software Engineer with experience in Python and machine learning."
    scores = score_resumes(job_description, '../../data/processed/resume_vectors.pkl', '../../data/processed/jd_vectors.pkl', 'models/best_model.pkl')

    for filename, score in scores.items():
        print(f"{filename}: {score:.4f}")