import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def evaluate_sklearn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def evaluate_lstm_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    X_test, _, _ = joblib.load('../../data/processed/resume_vectors.pkl')
    y_test = [1] * (len(X_test) // 2) + [0] * (len(X_test) // 2)  # Dummy labels

    models = ['models/logistic_regression.pkl', 'models/svm.pkl', 'models/random_forest.pkl', 'models/lstm_model.h5']
    best_model = None
    best_score = 0

    for model_file in models:
        if model_file.endswith('.pkl'):
            model = joblib.load(model_file)
            accuracy, precision, recall, f1 = evaluate_sklearn_model(model, X_test, y_test)
        else:
            model = load_model(model_file)
            accuracy, precision, recall, f1 = evaluate_lstm_model(model, X_test, y_test)
        
        score = f1  # Using F1 score as the selection metric
        if score > best_score:
            best_score = score
            best_model = model_file

    print(f"Best Model: {best_model} with F1 Score: {best_score}")
    joblib.dump(best_model, 'models/best_model.pkl')