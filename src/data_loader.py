
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Any


def Load_all_documents(file_paths: list[tuple[Path, str]]) -> List[Any]:
    """
    Load only the selected files and convert them
    into LangChain Document objects.

    Supported: PDF, TXT, MD
    """

    documents =[]
    print(f"[DEBUG] Selected files: {len(file_paths)}")
    

    for file_path, original_filename in file_paths:

        file_path = Path(file_path).resolve()

        if not file_path.exists():
            print(f"[WARNING] File not found: {file_path}")
            continue

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
                f"from {original_filename}"
            )

            for document in loaded:
                document.metadata["source"] = original_filename
                document.metadata["file_type"] = extension
                document.metadata["file_name"] = original_filename

            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            raise RuntimeError(f"Failed to load document: {file_path}") from e

    print(f"[DEBUG] Total documents loaded: {len(documents)}")

    return documents
