# src/app/app.py
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import joblib
from src.preprocessing.extract_text import extract_text_from_pdf
from src.preprocessing.preprocess import preprocess_text
from src.feature_extraction.vectorize import vectorize_text
from src.scoring.scoring import score_resumes
import PyPDF2 # type: ignore

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    job_description = request.form.get('job_description_text')
    job_description_file = request.files.get('job_description')
    resumes = request.files.getlist('resumes')

    if job_description_file and allowed_file(job_description_file.filename):
        job_description_path = secure_filename(job_description_file.filename)
        job_description_file.save(job_description_path)
        job_description = extract_text_from_pdf(job_description_path)
        os.remove(job_description_path)

    job_description = preprocess_text(job_description)

    for resume in resumes:
        if resume and allowed_file(resume.filename):
            resume_path = secure_filename(resume.filename)
            resume.save(os.path.join('../../data/raw/resumes/', resume_path))
            resume_text = extract_text_from_pdf(resume_path)
            os.remove(resume_path)
            resume_text = preprocess_text(resume_text)
            # Vectorize the resume text
            vectorize_text('../../data/processed/preprocessed_resumes', 'resume_vectors.pkl', [resume_text])
        else:
            return "Invalid file format for resume", 400

    # Vectorize the job description text
    vectorize_text('../../data/processed/preprocessed_job_descriptions', 'jd_vectors.pkl', [job_description])

    scores = score_resumes(job_description, '../../data/processed/resume_vectors.pkl', '../../data/processed/jd_vectors.pkl', 'models/best_model.pkl')

    return render_template('results.html', rankings=scores)

if __name__ == "__main__":
    app.run(debug=True)