class ChatHistory:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages

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

    def clear(self):
        self.messages.clear()