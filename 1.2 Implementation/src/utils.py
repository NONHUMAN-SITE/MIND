import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def merge_pdfs_text(pdfs_paths) -> str:
    try:
        text = ""
        for pdf_path in pdfs_paths:
            text += extract_text_from_pdf(pdf_path)
        return text
    except Exception as e:
        raise ValueError(f"Error merging pdfs: {e}. The pdfs paths must be valid paths to pdf files. The pdfs paths are: {pdfs_paths}")


def setup_experiment(config:dict,experiment_id:str):
    
    os.makedirs(config['paths']['experiments_dir'], exist_ok=True)
    
    experiment_dir = f"{config['paths']['experiments_dir']}/{experiment_id}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    models_dir = f"{experiment_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    tensorboard_dir = f"{experiment_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)

    return experiment_dir,models_dir,tensorboard_dir