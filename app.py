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
    test_questions = [
     "What example is given for FCFS?", "What are the execution steps in the FCFS example?", "Which scheduling algorithm is described as non-preemptive?", "What is the difference between preemptive and non-preemptive scheduling?", "Explain both FCFS and Round Robin.","What scheduling algorithms are listed in the document?", "Which scheduling algorithms are mentioned on page 6?", "Is Round Robin mentioned in the document?", "What does the document say about Round Robin?", "Explain FCFS scheduling."
]

    index = 0

    while index < len(test_questions):

        question = test_questions[index]

        print("\n" + "=" * 70)
        print(f"TEST {index + 1}")
        print(f"User: {question}")
        print("=" * 70)
        if rag.is_mention_question(question):
            print("[DEBUG] Mention question detected")
            term = rag.extract_mention_term(question)
            
            result = {
                    "answer": "Mention question detected",
                    "sources": []
                    }
            
        
        else:
            result = rag.Search_and_summarize(
                question,
                top_k=3
            )

        print("\nBot:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")

            for source in result["sources"]:
                print(
                    f"📄 {source['file_name']} | "
                    f"Page: {source['page']}"
                )

        index += 1