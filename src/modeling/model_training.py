from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_models(X, labels):
    # Train a logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X, labels)
    y_pred = logistic_model.predict(X)
    logistic_mse = mean_squared_error(labels, y_pred)
    print(f"Logistic Regression MSE: {logistic_mse:.4f}")

    # Train an SVM model
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X, labels)
    y_pred = svm_model.predict(X)
    svm_mse = mean_squared_error(labels, y_pred)
    print(f"SVM MSE: {svm_mse:.4f}")

    # Train a random forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X, labels)
    y_pred = rf_model.predict(X)
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

        # Check if the vectors are 0-dimensional arrays
        if resume_vectors.ndim == 0:
            resume_vectors = np.expand_dims(resume_vectors, axis=0)
        if jd_vectors.ndim == 0:
            jd_vectors = np.expand_dims(jd_vectors, axis=0)

        # Concatenate the resume and job description vectors
        X = np.concatenate([resume_vectors, jd_vectors], axis=0)

        # Compute the cosine similarity between each resume and each job description
        similarity_matrix = cosine_similarity(X)
        labels = np.max(similarity_matrix, axis=1)

        logistic_model, svm_model, rf_model = train_models(X, labels)

        # Save the trained models
        joblib.dump(logistic_model, 'models/logistic_model.pkl')
        joblib.dump(svm_model, 'models/svm_model.pkl')
        joblib.dump(rf_model, 'models/rf_model.pkl')
    except FileNotFoundError:
        print("Error: 'data/resume_vectors.pkl' or 'data/jd_vectors.pkl' file not found.")
    except Exception as e:
        print(f"Error: {e}")

