from src.data_loader import Load_all_documents
from src.chat_service import ChatService
from pathlib import Path

def run_demo():
    backend = ChatService()
    BASE_DIR = Path(__file__).resolve().parent
    choice = int(input("Please Enter your Input: \n New Chat(1), Existing Chat(2) "))
    selected_files = []
    match choice:
        case 1:
            chat_id = backend.create_chat()
            if not maybe_upload_files(backend, chat_id, "Upload files for this new chat?"):
                print("Please upload atleast a file")
                return
        case 2:
            chat_id = input("Enter existing Chat ID: ").strip()
            chatid_DIR = BASE_DIR / "chat_data"/chat_id
            if not chatid_DIR.is_dir():
                print("Chat not found.")
                return
            maybe_upload_files(backend, chat_id, "Upload extra files to this chat?")
        case _:
            print("Invalid choice")
            return
    
    
    
    print(f"Chat ID: {chat_id}")
    

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

def get_user_files():

    selected_files = []

    print("\nEnter PDF file paths.")
    print("Type 'done' when finished.\n")

    while True:

        file_path = input("File path: ").strip()

        if file_path.lower() == "done":
            break

        if not file_path:
            print("Please enter a file path.")
            continue

        path = Path(file_path).expanduser()

        if not path.exists() or not path.is_file():
            print("File does not exist. Try again.")
            continue

        if path.suffix.lower() != ".pdf":
            print("Only PDF files are supported.")
            continue

        if path.stat().st_size == 0:
            print("Empty files are not allowed.")
            continue

        if path in selected_files:
            print("File already selected.")
            continue

        selected_files.append(path)
        print(f"Added: {path.name}")

    return selected_files

def maybe_upload_files(backend, chat_id, prompt):
    answer = input(prompt + " (y/n): ").strip().lower()

    if answer not in {"y", "yes"}:
        return False    

    selected_files = get_user_files()
    if not selected_files:
        print("No files selected.")
        return False

    print(backend.upload_documents(chat_id, selected_files))
    return True

if __name__ == "__main__":
    run_demo()
    