import pickle
import os
from typing import List, Any
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embedding import EmbeddingPipeline
from src.reranker import Reranker
import numpy as np

class Faissvectorestore:
    def __init__(self, persist_dir: str="faiss_store", embedding_model: str = "all-MiniLM-L6-v2", CrossEncoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", chunk_size : int=1000, chunk_overlap: int= 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.CrossEncoder_model = CrossEncoder_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        reranker_model = CrossEncoder("BAAI/bge-reranker-base")
        self.reranker = Reranker(reranker_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents ... ")
        emb_pipe = EmbeddingPipeline(model=self.model, chunk_size=self.chunk_size, chunk_overlap= self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
        {
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", 0) + 1,
            "file_name": chunk.metadata.get("source")
        }
        for chunk in chunks
        ]
        if len(embeddings) != len(metadatas):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) does not match "
                f"metadata count ({len(metadatas)})."
            )
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")


    def add_embeddings(self, embeddings: np.array, metadatas: List[Any]= None):
        if len(embeddings) != len(metadatas):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) does not match "
                f"metadata count ({len(metadatas)})."
            )
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        elif self.index.d != dim:
            raise ValueError(
                f"Dimension mismatch! FAISS index expects {self.index.d} dimensions, "
                f"but embeddings have {dim} dimensions."
            )
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Integrity check failed! Index total ({self.index.ntotal}) "
                f"does not match metadata length ({len(self.metadata)})."
            )
        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS db index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Vector store files not found in {self.persist_dir}"
            )

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        print("self.index:", self.index)
        print("query_embedding shape:", query_embedding.shape)
        
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            # Prevent FAISS -1 sentinel value and out-of-bounds negative indexing
            if idx != -1 and 0 <= idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, question: str, top_k: int = 5):

        print(f"[INFO] Querying vector store for: '{question}'")
    
        candidate_k = max(top_k * 3, 10)
    
        query_emb = self.model.encode(
            [question]
        ).astype("float32")
    
        results = self.search(
            query_emb,
            top_k=candidate_k
        )
    
        reranked_results = self.reranker.rerank(
            question,
            results,
            top_k=top_k
        )
    
        return reranked_results

    def exists(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        return os.path.exists(faiss_path) and os.path.exists(meta_path)