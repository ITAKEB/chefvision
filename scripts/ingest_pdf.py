import sys
from pathlib import Path
from services.ingestion import ingest_pdf

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "backend"))
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def main(): 
    """
        Recorrer todos los PDFs, extrae el texto 
        y lo guarda en chromaDB
    """
    pdfs = list(RAW_DATA_DIR.glob("*.pdf"))
    # all_docs = []

    for pdf_path in pdfs:
        base_name = pdf_path.stem
        # pdf_chunk = ingest_pdf(pdf_path)
        print(f"Procesando {pdf_path.name}...")
        
if __name__ == "__main__":
    main()





