
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Any


def Load_all_documents(data_dir : str) -> List[Any]:
    """
    Load all supported  files from the directory and convert to LANGChain Supported structure
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """

    # Use Project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents =[]

    #PDF Files 

    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files in {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF:{pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)}  PDF Docs from {pdf_file}")
            documents.extend(loaded)
            print(f"Loaded {len(documents)} documents from {pdf_file}")
        except Exception as e:
            print(f"[ERROR] Failed to Load PDF {pdf_file}: {e}")

    return documents
