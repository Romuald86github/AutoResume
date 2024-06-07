import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def vectorize_text(input_dir, output_file):
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r') as file:
                documents.append(file.read())
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    with open(output_file, 'wb') as file:
        pickle.dump((vectorizer, vectors), file)

if __name__ == "__main__":
    vectorize_text('../../data/processed/resumes', 'resume_vectors.pkl')
    vectorize_text('../../data/processed/job_descriptions', 'jd_vectors.pkl')