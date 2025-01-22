import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def merge_pdfs_text(pdf_paths) -> str:
    try:
        text = ""
        for pdf_path in pdf_paths:
            text += extract_text_from_pdf(pdf_path)
        return text
    except Exception as e:
        raise ValueError(f"Error merging pdfs: {e}. The pdfs paths must be valid paths to pdf files. The pdfs paths are: {pdf_paths}")

