from src.data_loader import Load_all_documents

from src.search import RAGSearch
from src.vectorstore import Faissvectorestore
from src.chat_history import ChatHistory


if __name__ == "__main__":
    #docs = Load_all_documents("data")
    store = Faissvectorestore("faiss_store")
    chat_history = ChatHistory(max_messages=10)
    #store.build_from_documents(docs)
    store.load()
    
    rag = RAGSearch(store, chat_history)
    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Chat ended.")
            break

        result = rag.Search_and_summarize(question)

        print("Bot:", result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(
                f"📄 {source['file_name']} | "
                f"Page: {source['page']}"
            )
       