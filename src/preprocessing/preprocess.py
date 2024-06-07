import os
import re
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def preprocess_text(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r') as file:
                text = file.read()
            clean = clean_text(text)
            tokens = tokenize_and_remove_stopwords(clean)
            with open(os.path.join(output_dir, filename), 'w') as text_file:
                text_file.write(' '.join(tokens))

if __name__ == "__main__":
    preprocess_text('../../data/processed/resumes', '../../data/processed/resumes')
    preprocess_text('../../data/processed/job_descriptions', '../../data/processed/job_descriptions')