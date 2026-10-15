from src.data_loader import Load_all_documents
from src.chat_service import ChatService


def run_demo():
    backend = ChatService()

    chat_id = backend.create_chat()
    print(f"Chat ID: {chat_id}")
    documents = Load_all_documents("data")
    backend.add_documents(chat_id, documents)
    while True:
        question = input("\nYou (or type /exit): ").strip()

        if question.lower() == "/exit":
            break

        if not question:
            continue

        result = backend.ask(chat_id, question)
        print(f"\nAssistant: {result['answer']}")

        for source in result["sources"]:
            print(f"Source: {source['file_name']} | Page: {source['page']}")

            
if __name__ == "__main__":
    run_demo()
    