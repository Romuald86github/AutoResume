import os
import PyPDF2

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def save_extracted_text(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(file_path)
            with open(os.path.join(output_dir, filename.replace('.pdf', '.txt')), 'w') as text_file:
                text_file.write(text)

if __name__ == "__main__":
    save_extracted_text('data/raw/resumes', 'data/processed/resumes')
    save_extracted_text('data/raw/job_descriptions', 'data/processed/job_descriptions')