from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from src.chat_history import ChatHistory
from src.search import RAGSearch
from src.vectorstore import Faissvectorestore

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
            history = ChatHistory(max_messages=10)
            self.rags[chat_id] = RAGSearch(store, history)

        return self.rags[chat_id].Search_and_summarize(question, top_k=3)

