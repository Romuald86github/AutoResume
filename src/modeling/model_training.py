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
    # Split the data into training and testing sets
    train_size = int(0.8 * len(resume_vectors))
    X_train, X_test = resume_vectors[:train_size], resume_vectors[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    # Train a logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    logistic_mse = mean_squared_error(y_test, y_pred)
    print(f"Logistic Regression MSE: {logistic_mse:.4f}")

    # Train an SVM model
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    svm_mse = mean_squared_error(y_test, y_pred)
    print(f"SVM MSE: {svm_mse:.4f}")

    # Train a random forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest MSE: {rf_mse:.4f}")

    return logistic_model, svm_model, rf_model

if __name__ == "__main__":
    try:
        # URL of the file in your GitHub repository
        url = 'https://raw.githubusercontent.com/Romuald86github/AutoResume/main/data/vectors.pkl'

        # Download the file from the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Load the data from the downloaded file
            data = joblib.load(io.BytesIO(response.content))

            resume_vectors = data['resume_vectors']
            jd_vectors = data['jd_vectors']

            print("resume_vectors shape:", resume_vectors.shape)
            print("jd_vectors shape:", jd_vectors.shape)

            # Compute the cosine similarity between each resume and each job description
            similarity_matrix = cosine_similarity(resume_vectors, jd_vectors.T)

            # Find the highest similarity score for each job description and use it as the label
            labels = np.max(similarity_matrix, axis=0)

            logistic_model, svm_model, rf_model = train_models(resume_vectors, jd_vectors, labels)
        else:
            print(f"Error: Failed to download the file. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")