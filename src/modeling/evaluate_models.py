import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_models(resume_vectors, jd_vectors, resume_filenames, jd_filenames):
    # Load the trained models
    logistic_model = joblib.load('models/logistic_model.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    rf_model = joblib.load('models/rf_model.pkl')

    # Evaluate the models
    logistic_mse = mean_squared_error(logistic_model.predict(resume_vectors), logistic_model.predict(jd_vectors))
    svm_mse = mean_squared_error(svm_model.predict(resume_vectors), svm_model.predict(jd_vectors))
    rf_mse = mean_squared_error(rf_model.predict(resume_vectors), rf_model.predict(jd_vectors))

    print(f"Logistic Regression MSE: {logistic_mse:.4f}")
    print(f"SVM MSE: {svm_mse:.4f}")
    print(f"Random Forest MSE: {rf_mse:.4f}")

    # Select the best model
    best_model_name = min((logistic_mse, 'logistic_model.pkl'), (svm_mse, 'svm_model.pkl'), (rf_mse, 'rf_model.pkl'), key=lambda x: x[0])[1]
    best_model = joblib.load(f'models/{best_model_name}')

    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')

    return best_model

if __name__ == "__main__":
    data = joblib.load('data/vectors.pkl')
    best_model = evaluate_models(data['resume_vectors'], data['jd_vectors'], data['resume_filenames'], data['jd_filenames'])