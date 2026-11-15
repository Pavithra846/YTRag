import json
from pathlib import Path
from typing import Any, List
from uuid import UUID, uuid4

from src.chat_history import ChatHistory
from src.search import RAGSearch
from src.vectorstore import Faissvectorestore
from src.data_loader import Load_all_documents

import shutil
import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

class ChatService:
    def __init__(self, storage_dir: str = "chat_data"):
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.stores: dict[str, Faissvectorestore] = {}
        self.rags: dict[str, RAGSearch] = {}

    def _normalize_chat_id(self, chat_id: str) -> str:
        # Validating the ID also prevents it being used as a path traversal string.
        return str(UUID(chat_id))

    def create_chat(self) -> str:
        chat_id = str(uuid4())
        chat_dir = self.storage_dir / chat_id
        (chat_dir / "uploads").mkdir(parents=True)
        return chat_id
    
    def _get_store(
        self,
        chat_id: str,
        require_existing_index: bool = False,
    ) -> Faissvectorestore:
        chat_id = self._normalize_chat_id(chat_id)

        if chat_id in self.stores:
            store = self.stores[chat_id]
        else:
            store_dir = self.storage_dir / chat_id / "faiss_store"
            store = Faissvectorestore(persist_dir=str(store_dir))

            if store.exists():
                store.load()
            elif require_existing_index:
                raise FileNotFoundError(
                    f"No saved vector store for chat {chat_id}"
                )

            self.stores[chat_id] = store

        if require_existing_index and store.index is None:
            raise FileNotFoundError(
                f"No documents have been indexed for chat {chat_id}"
            )

        return store

    def add_documents(self, chat_id: str, documents: list[Any]) -> None:
        if not documents:
            raise ValueError("No documents were provided.")

        store = self._get_store(chat_id)
        # Pass only newly uploaded documents, not every old upload again.
        store.build_from_documents(documents)

        # If this chat's RAG wrapper already exists, it uses the updated store.
        self.rags.pop(self._normalize_chat_id(chat_id), None)

    def load_chat(self, chat_id: str) -> None:
        chat_id = self._normalize_chat_id(chat_id)
        self._get_store(chat_id, require_existing_index=True)

    def ask(self, chat_id: str, question: str) -> dict:
        chat_id = self._normalize_chat_id(chat_id)

        if chat_id not in self.rags:
            store = self._get_store(chat_id, require_existing_index=True)
            chat_dir = self.storage_dir / chat_id

            history_file = chat_dir / "chat_history.json"
            
            history = ChatHistory(
                max_messages=10,
                history_file=history_file
            )
            self.rags[chat_id] = RAGSearch(store, history)

        return self.rags[chat_id].Search_and_summarize(question, top_k=3)

    def upload_documents(self, chat_id: str,  uploaded_files: List[Path]):

        chat_id = self._normalize_chat_id(chat_id)
    
        upload_dir = self.storage_dir / chat_id / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.storage_dir / chat_id  / "uploaded_files.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    indexed_files = json.load(f)

                if not isinstance(indexed_files, dict):
                    raise ValueError("Invalid uploaded files manifest.")

            except json.JSONDecodeError as e:
                raise ValueError(
                    "The uploaded files manifest is corrupted."
                ) from e

        else:
            indexed_files = {}

        new_files = []
        new_records = {}
        pending_hashes = set()
        copied_files = []

        for file_path in uploaded_files:

            file_path = Path(file_path)

            if not file_path.is_file():
                print(f"[WARNING] File not found: {file_path}")
                continue

            if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
                print(f"[WARNING] Unsupported file: {file_path.name}")
                continue

            # Calculate content hash
            file_hash = self.get_file_hash(file_path)

            # Check whether identical content was already indexed
            if file_hash in indexed_files or file_hash in pending_hashes:
                print(f"[INFO] Duplicate file skipped: {file_path.name}")
                continue

            destination = upload_dir / file_path.name

            if destination.exists():
                if self.get_file_hash(destination) != file_hash:
                    raise FileExistsError(
                        f"A different file named {file_path.name} "
                        f"already exists in this chat's uploads."
                    )
            else:
                shutil.copy2(file_path, destination)
                copied_files.append(destination)

            new_files.append(destination)
            new_records[file_hash] = {
                "file_name": destination.name,
                "original_name": file_path.name
            }
            pending_hashes.add(file_hash)
        if not new_files:
            return "No new files to upload."
        
        try:
            # Load only newly uploaded files
            documents = Load_all_documents(new_files)

            if not documents:
                raise ValueError("No documents found.")

            self.add_documents(chat_id, documents)

            indexed_files.update(new_records)

            temp_manifest = manifest_path.with_suffix(".tmp")

            with open(temp_manifest, "w") as f:
                json.dump(indexed_files, f, indent=4)
            
            temp_manifest.replace(manifest_path)

        except Exception:
            # Remove files copied during this upload attempt
            for file_path in copied_files:
                if file_path.exists():
                    file_path.unlink()

            raise

        return f"Successfully indexed {len(new_files)} new files."


    def get_file_hash(self, file_path: Path) -> str:

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()