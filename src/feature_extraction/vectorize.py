import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src.preprocessing.preprocess import preprocess_text

def vectorize_text(input_dir, output_file):
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read()
            preprocessed_text = preprocess_text(text)
            documents.append(preprocessed_text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    with open(output_file, 'wb') as file:
        pickle.dump((vectorizer, vectors), file)

if __name__ == "__main__":
    vectorize_text('data/processed/preprocessed_resumes', 'data/resume_vectors.pkl')
    vectorize_text('data/processed/preprocessed_job_descriptions', 'data/job_description_vectors.pkl')