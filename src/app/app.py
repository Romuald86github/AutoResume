import os
import joblib
import numpy as np
import PyPDF2
import re
import nltk
import string
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def remove_punctuation(text):
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation from the text
    text_without_punctuation = text.translate(translator)
    
    return text_without_punctuation

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = remove_punctuation(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join the filtered tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user-uploaded resume and job description
        resume_file = request.files['resume']
        jd_file = request.files['job_description']

        # Save the uploaded files
        resume_filename = secure_filename(resume_file.filename)
        jd_filename = secure_filename(jd_file.filename)
        resume_file.save(os.path.join('uploads', resume_filename))
        jd_file.save(os.path.join('uploads', jd_filename))

        # Load the required data and models
        data = joblib.load('data/vectors.pkl')
        resume_vectorizer = data['resume_vectorizer']
        jd_vectorizer = data['jd_vectorizer']
        best_model = joblib.load('models/best_model.pkl')

        # Extract and preprocess the text from the uploaded files
        user_resume_text = preprocess_text(extract_text_from_pdf(os.path.join('uploads', resume_filename)))
        user_jd_text = preprocess_text(extract_text_from_pdf(os.path.join('uploads', jd_filename)))

        # Transform the user-uploaded data using the vectorizers
        user_resume_vector = resume_vectorizer.transform([user_resume_text])
        user_jd_vector = jd_vectorizer.transform([user_jd_text])

        # Compute the cosine similarity score
        user_similarity_score = cosine_similarity(user_resume_vector, user_jd_vector.T)[0][0]

        # Use the best model to predict the similarity
        prediction = best_model.predict([[user_similarity_score]])

        return render_template('index.html', prediction=prediction[0])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)