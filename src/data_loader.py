
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
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

    supported_files = []

    supported_files.extend(data_path.glob("**/*.pdf"))
    supported_files.extend(data_path.glob("**/*.txt"))
    supported_files.extend(data_path.glob("**/*.md"))
    print(f"[DEBUG] Found {len(supported_files)} supported files")

    for file_path in supported_files:
        print(f"[DEBUG] Loading PDF:{file_path}")
        try:
            extension = file_path.suffix.lower()

            # Select loader based on file type
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif extension == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            elif extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))

            else:
                print(f"[WARNING] Unsupported file: {file_path}")
                continue
            loaded = loader.load()
            print(
                f"[DEBUG] Loaded {len(loaded)} documents "
                f"from {file_path.name}"
            )
             # Add source metadata
            for document in loaded:
                document.metadata["source"] = file_path.name
                document.metadata["file_type"] = extension
            documents.extend(loaded)
        except Exception as e:
            print(
                f"[ERROR] Failed to load "
                f"{file_path}: {e}"
            )
        
    print(
        f"[DEBUG] Total documents loaded: "
        f"{len(documents)}"
    )

    return documents
