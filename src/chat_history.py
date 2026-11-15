import json
from pathlib import Path


class ChatHistory:
    def __init__(self, max_messages=10, history_file=None):
        self.messages = []
        self.max_messages = max_messages
        self.history_file = Path(history_file) if history_file else None
        if self.history_file and self.history_file.exists():
            self.load_history()

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })
        self._trim()
        

    def add_assistant_message(self, message):
        self.messages.append({
            "role": "assistant",
            "content": message
        })
        self._trim()

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_history(self):
        return self.messages

    def get_formatted_history(self):
        history = ""

        for message in self.messages:
            role = message["role"].capitalize()
            content = message["content"]

            history += f"{role}: {content}\n"

        return history
    def save_history(self):
        if not self.history_file:
            return

        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        with self.history_file.open("w", encoding="utf-8") as file:
            json.dump(
                {"messages": self.messages},
                file,
                indent=4,
                ensure_ascii=False
            )
    def load_history(self):
        if not self.history_file or not self.history_file.exists():
            return

        with self.history_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.messages = data.get("messages", [])

    def clear(self):
        self.messages.clear()
        self.save_history()
