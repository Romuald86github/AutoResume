import os
import re
import nltk
import string
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

def preprocess_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read()

            preprocessed_text = preprocess_text(text)

            output_file_path = os.path.join(output_dir, filename)
            with open(output_file_path, 'w') as output_file:
                output_file.write(preprocessed_text)

if __name__ == "__main__":
    preprocess_files('data/processed/resumes', 'data/processed/preprocessed_resumes')
    preprocess_files('data/processed/job_descriptions', 'data/processed/preprocessed_job_descriptions')