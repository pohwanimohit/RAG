from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def process_pdf(pdf_path: str):
   
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"\n[PDF] Loading: {pdf_path.name}")

    pdf_loader = PyPDFLoader(str(pdf_path))
    pdf_documents = pdf_loader.load()

    print(f"[PDF] Total pages loaded: {len(pdf_documents)}")

    for doc in pdf_documents:
        print(f"\nPage: {doc.metadata.get('page')}")
        print(doc.page_content[:100])
        print("-" * 30)

    return pdf_documents