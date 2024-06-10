import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def vectorize_text(input_dir, output_file):
    resume_texts = []
    resume_filenames = []
    jd_texts = []
    jd_filenames = []

    logging.info(f"Processing files in directory: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read().strip()

            if text:
                if filename.startswith("resume_") or "resume" in filename.lower():
                    logging.info(f"Found resume file: {filename}")
                    resume_texts.append(text)
                    resume_filenames.append(filename)
                elif filename.startswith("job_") or "job" in filename.lower():
                    logging.info(f"Found job description file: {filename}")
                    jd_texts.append(text)
                    jd_filenames.append(filename)

    print(f"Length of resume_texts: {len(resume_texts)}")
    print(f"Length of jd_texts: {len(jd_texts)}")

    if resume_texts or jd_texts:
        logging.info("Processing resume and job description vectors...")
        if resume_texts:
            resume_vectorizer = TfidfVectorizer()
            resume_vectors = resume_vectorizer.fit_transform(resume_texts)
            logging.info(f"Resume vectors shape: {resume_vectors.shape}")
        else:
            logging.warning("No resume texts found.")
            resume_vectors = None
            resume_vectorizer = None

        if jd_texts:
            jd_vectorizer = TfidfVectorizer()
            jd_vectors = jd_vectorizer.fit_transform(jd_texts)
            logging.info(f"Job description vectors shape: {jd_vectors.shape}")
        else:
            logging.warning("No job description texts found.")
            jd_vectors = None
            jd_vectorizer = None

        logging.info(f"Saving output to: {output_file}")
        try:
            with open(output_file, 'wb') as file:
                joblib.dump({
                    'resume_texts': resume_texts,
                    'resume_vectors': resume_vectors,
                    'resume_filenames': resume_filenames,
                    'jd_texts': jd_texts,
                    'jd_vectors': jd_vectors,
                    'jd_filenames': jd_filenames,
                    'resume_vectorizer': resume_vectorizer,
                    'jd_vectorizer': jd_vectorizer
                }, file)
            logging.info("Output saved successfully.")
        except Exception as e:
            logging.error(f"Error saving output: {e}")
    else:
        logging.warning("No valid text files found in the input directories.")

if __name__ == "__main__":
    vectorize_text('data/processed/preprocessed_resumes', 'data/vectors.pkl')
    vectorize_text('data/processed/preprocessed_job_descriptions', 'data/vectors.pkl')