def vectorize_text(input_dir, output_file):
    resume_texts = []
    resume_filenames = []
    jd_texts = []
    jd_filenames = []

    print(f"Processing files in directory: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read().strip()

            if text:
                if filename.startswith("resume_") or "resume" in filename.lower():
                    print(f"Found resume file: {filename}")
                    resume_texts.append(text)
                    resume_filenames.append(filename)
                elif filename.startswith("job_") or "job" in filename.lower():
                    print(f"Found job description file: {filename}")
                    jd_texts.append(text)
                    jd_filenames.append(filename)

    if resume_texts and jd_texts:
        print("Processing resume and job description vectors...")
        resume_vectorizer = TfidfVectorizer()
        resume_vectors = resume_vectorizer.fit_transform(resume_texts)

        jd_vectorizer = TfidfVectorizer()
        jd_vectors = jd_vectorizer.fit_transform(jd_texts)

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
    else:
        print("No valid text files found in the input directories.")

if __name__ == "__main__":
    vectorize_text('data/processed/preprocessed_resumes', 'data/vectors.pkl')
    vectorize_text('data/processed/preprocessed_job_descriptions', 'data/vectors.pkl')