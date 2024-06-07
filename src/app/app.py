from flask import Flask, request, render_template
import os
from src.scoring.scoring import score_resumes

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    job_description = request.form['job_description']
    resumes = request.files.getlist('resumes')

    for resume in resumes:
        resume.save(os.path.join('../../data/raw/resumes/', resume.filename))
    
    # Extract and preprocess resumes
    os.system('python src/preprocessing/extract_text.py')
    os.system('python src/preprocessing/preprocess.py')
    os.system('python src/feature_extraction/vectorize.py')
    
    scores = score_resumes(job_description, '../../data/processed/resume_vectors.pkl', '../../data/processed/jd_vectors.pkl', 'models/best_model.pkl')
    
    return render_template('results.html', rankings=scores)
