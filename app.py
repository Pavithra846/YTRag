from src.data_loader import Load_all_documents

from src.search import RAGSearch
from src.vectorstore import Faissvectorestore
from src.chat_history import ChatHistory


if __name__ == "__main__":
    
    store = Faissvectorestore("faiss_store")
    if store.exists():
        store.load()
    else:
        print("[INFO] No vector store found. Building vector store...")
        docs = Load_all_documents("data")
        if not docs:
            raise ValueError("No documents found in the data directory.")
        
        store.build_from_documents(docs)

        
    chat_history = ChatHistory(max_messages=10)
    
    
    rag = RAGSearch(store, chat_history)
    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Chat ended.")
            break

        result = rag.Search_and_summarize(question, top_k=15)

        print("Bot:", result["answer"])
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"📄 {source['file_name']} | "
                    f"Page: {source['page']}"
                )
       