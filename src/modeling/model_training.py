from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_models(resume_vectors, jd_vectors, labels):
    # Ensure the number of dimensions in resume_vectors and jd_vectors are the same
    if resume_vectors.ndim != jd_vectors.ndim:
        if resume_vectors.ndim == 1:
            resume_vectors = resume_vectors[:, np.newaxis]
        else:
            jd_vectors = jd_vectors[:, np.newaxis]

    # Ensure the number of features in resume_vectors and jd_vectors are the same
    max_features = max(resume_vectors.shape[1], jd_vectors.shape[1])
    if resume_vectors.shape[1] < max_features:
        resume_vectors = np.hstack((resume_vectors, np.zeros((resume_vectors.shape[0], max_features - resume_vectors.shape[1]))))
    if jd_vectors.shape[1] < max_features:
        jd_vectors = np.hstack((jd_vectors, np.zeros((jd_vectors.shape[0], max_features - jd_vectors.shape[1]))))

    # Train a logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(resume_vectors, labels)
    y_pred = logistic_model.predict(resume_vectors)
    logistic_mse = mean_squared_error(labels, y_pred)
    print(f"Logistic Regression MSE: {logistic_mse:.4f}")

    # Train an SVM model
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(resume_vectors, labels)
    y_pred = svm_model.predict(resume_vectors)
    svm_mse = mean_squared_error(labels, y_pred)
    print(f"SVM MSE: {svm_mse:.4f}")

    # Train a random forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(resume_vectors, labels)
    y_pred = rf_model.predict(resume_vectors)
    rf_mse = mean_squared_error(labels, y_pred)
    print(f"Random Forest MSE: {rf_mse:.4f}")

    return logistic_model, svm_model, rf_model

if __name__ == "__main__":
    try:
        # Load the data from the saved files
        resume_data = joblib.load('data/resume_vectors.pkl')
        jd_data = joblib.load('data/jd_vectors.pkl')

        resume_vectors = resume_data['resume_vectors']
        jd_vectors = jd_data['jd_vectors']

        # Ensure the number of dimensions in resume_vectors and jd_vectors are the same
        if resume_vectors.ndim != jd_vectors.ndim:
            if resume_vectors.ndim == 1:
                resume_vectors = resume_vectors[:, np.newaxis]
            else:
                jd_vectors = jd_vectors[:, np.newaxis]

        # Ensure the number of features in resume_vectors and jd_vectors are the same
        max_features = max(resume_vectors.shape[1], jd_vectors.shape[1])
        if resume_vectors.shape[1] < max_features:
            resume_vectors = np.hstack((resume_vectors, np.zeros((resume_vectors.shape[0], max_features - resume_vectors.shape[1]))))
        if jd_vectors.shape[1] < max_features:
            jd_vectors = np.hstack((jd_vectors, np.zeros((jd_vectors.shape[0], max_features - jd_vectors.shape[1]))))

        # Compute the cosine similarity between each resume and each job description
        similarity_matrix = cosine_similarity(resume_vectors, jd_vectors)
        labels = np.max(similarity_matrix, axis=1)

        logistic_model, svm_model, rf_model = train_models(resume_vectors, jd_vectors, labels)

        # Save the trained models
        joblib.dump(logistic_model, 'models/logistic_model.pkl')
        joblib.dump(svm_model, 'models/svm_model.pkl')
        joblib.dump(rf_model, 'models/rf_model.pkl')
    except FileNotFoundError:
        print("Error: 'data/resume_vectors.pkl' or 'data/jd_vectors.pkl' file not found.")
    except Exception as e:
        print(f"Error: {e}")