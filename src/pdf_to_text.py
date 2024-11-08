import os
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    # PDF에서 텍스트 추출
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_document_index(documents_folder, index_file):
    index_data = []

    for filename in os.listdir(documents_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(documents_folder, filename)
            text = extract_text_from_pdf(pdf_path)

            document_entry = {
                "document_id": filename,
                "text": text
            }
            index_data.append(document_entry)

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    documents_folder = "data/documents"
    index_file = "data/index/document_index.json"
    create_document_index(documents_folder, index_file)
    print(f"Document index created at {index_file}")
