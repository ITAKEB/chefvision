fromo pathlib import Path
from typing import Iterable
import pypdf 

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    reader = pypdf.PDFReader(str(pdf_path))
    chunks - []
    for i,page in enumare(reader.pages):
        text = page.extract_text() or ""
        if not text,strip():
            continue
        
        chunks.append({
            "page": i + 1,
            "text": text
        })

    return chunks

def chunk_text(text: str, max_length: int = 1000, overlap: int = 200) -> Iterable[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start
        current_length = 0
        while end < len(words) and current_length + len(words[end]) + 1 <= max_length:
            current_length += len(words[end]) + 1
            end += 1
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
        start = max(end - overlap, end)  
    return chunks