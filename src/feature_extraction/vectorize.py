import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(input_dir, output_file):
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read()
            documents.append(text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    with open(output_file, 'wb') as file:
        joblib.dump({'raw_texts': documents, 'vectorizer': vectorizer}, file)

if __name__ == "__main__":
    vectorize_text('data/processed/preprocessed_resumes', 'data/resume_vectors.pkl')
    vectorize_text('data/processed/preprocessed_job_descriptions', 'data/job_description_vectors.pkl')