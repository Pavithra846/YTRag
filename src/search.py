from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import re

load_dotenv()
import json

class RAGSearch:
    def __init__(self, Vectorstore, chat_history, llm_model: str = "openai/gpt-oss-20b"):
        self.vectorstore = Vectorstore
        self.chat_history = chat_history
        self.conversation_state = {
        "active_topics": [],
        "available_topics": [],
        "unavailable_topics": [],
        "last_topic": None
        }
        GROQ_API_KEY  = os.getenv("GROQ_API_KEY")    
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY , model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def Search_and_summarize(self, query: str, top_k =3):
        #reteriver the context
        
        if not self.needs_query_analysis(query):
            questions = [query]
            ambiguous = False

        else:
            rewrite_prompt = ChatPromptTemplate.from_messages([
            (
            "system",
            """
            You are a query rewriting assistant for a PDF-based RAG system.

            Your task is to rewrite the user's current question into clear,
            standalone question(s) using the previous conversation and
            conversation state.

            Rules:

            1. Use the previous conversation to understand references such as
               "it", "this", "that", "they", "its", "each", "both", etc.

            2. Use the conversation state to understand which topics are currently
               active and whether those topics have relevant information available
               in the PDF.

            3. If the current question is already clear and standalone,
               return it unchanged.

            4. If the user introduces a new topic or concept, do NOT connect it
               to the previous conversation. Treat it as a new standalone question.

            5. If the current question contains a reference such as "it", "its",
               "this", "that", "they", "each", or "both", resolve the reference
               using the previous conversation and conversation state.

            6. If the previous user question contained multiple topics, do not
               automatically assume that a reference refers only to the last topic.

            7. For words such as "each" or "both", identify all entities that
               the user is referring to and create separate questions when needed.

               Example:
               Previous question:
               "What is the difference between binary and counting semaphores?"

               Current question:
               "Give an example of each."

               Output:
               {{
                   "ambiguous": false,
                   "questions": [
                       "Give an example of a binary semaphore.",
                       "Give an example of a counting semaphore."
                   ]
               }}

            8. If the conversation contains:
               "What is operating system and what is chunking?"

               and the conversation state indicates that operating system is
               available in the PDF while chunking is unavailable, then:

               Current question:
               "What is its usage?"

               should resolve "its" to the relevant supported topic:
               "operating system".

            9. If a reference has multiple genuinely possible meanings and
               conversation history and state cannot determine the intended
               meaning, DO NOT GUESS.

            10. If a reference is genuinely ambiguous, set "ambiguous" to true
                and return an empty questions array.

            11. If the current query contains multiple independent questions,
                put each question as a separate item in the "questions" array.

            12. Do NOT answer the question.

            13. Do NOT add information from your own knowledge.

            14. Do NOT use web search or internet information.

            15. Return ONLY valid JSON.

            16. The JSON must follow exactly this format:

            {{
                "ambiguous": false,
                "questions": [
                    "standalone question 1"
                ]
            }}

            17. For an ambiguous query, return:

            {{
                "ambiguous": true,
                "questions": []
            }}

            18. Do not add explanations outside the JSON.
            """
            ),  
        (   
            "human",
            """
            Previous conversation:
            {chat_history}

            Conversation state:
            {conversation_state}

            Current question:
            {query}

            Analyze the current question and return ONLY the required JSON.
            """
            )
            ])  
            state_text = f"""
                Active topics: {self.conversation_state["active_topics"]}
                Available topics: {self.conversation_state["available_topics"]}
                Unavailable topics: {self.conversation_state["unavailable_topics"]}
                Last topic: {self.conversation_state["last_topic"]}
                """
            formatted_prompt = rewrite_prompt.invoke({
                "chat_history": self.chat_history.get_formatted_history(),
                "conversation_state": state_text,
                "query": query
            })
        
            response = self.llm.invoke(formatted_prompt)

            try:
                result = json.loads(response.content.strip())

                ambiguous = result.get("ambiguous", False)
                questions = result.get("questions", [])

            except json.JSONDecodeError:
                # Safe fallback
                ambiguous = False
                questions = [query]
            # Validate questions
            if not isinstance(questions, list):
                questions = [query]

            questions = [
                q.strip()
                for q in questions
                if isinstance(q, str) and q.strip()
            ]
        if ambiguous:
            return {
                "answer": "Your question is ambiguous. Could you clarify what 'it' refers to?",
                "sources": []
            }   
        if not questions:
            questions = [query]
        # --------------------------------------------------
        # 3. Retrieve results for each question
        # --------------------------------------------------
        available_questions = []
        unavailable_questions = []
        all_results = []
        for question in questions:
            results = self.vectorstore.query(
                question,
                top_k=top_k
            )
            print(f"[DEBUG] Question: {question}")
            print(f"[DEBUG] Retrieved results: {len(results)}")
            for i, r in enumerate(results):
                print(
                    f"[DEBUG] Result {i+1}: "
                    f"page={r['metadata'].get('page')} "
                    f"score={r.get('score')}"
                )
            if results:
                available_questions.append(question)
                all_results.extend(results)
            else:
                unavailable_questions.append(question)

        self.update_conversation_state(
            available_questions,
            available=True
                )

        self.update_conversation_state(
            unavailable_questions,
            available=False
            )
        # --------------------------------------------------
        # 4. Build context
        # --------------------------------------------------

        texts = [r["metadata"].get("text", "") for r in all_results if r["metadata"]]
        sources = [
        {
            "page": r["metadata"].get("page"),
            "file_name": r["metadata"].get("file_name")
        }
        for r in all_results
        if r["metadata"]
        ]
        context = "\n\n".join(
        text for text in texts if text.strip()
)   
        
        if not context:
            return {
                    "answer": "The answer is not available in the provided documents.",
                    "sources": []
                    }
        ## Generate the answer uisng Groq LLM


        answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a PDF-based question-answering assistant.

            Your knowledge is restricted to the PDF context provided by the user.

            Rules:
            1. Answer the question using only information supported by the PDF context.
            2. You may combine information from multiple relevant parts of the context
               to form a clear answer.
            3. You may summarize or rephrase information from the PDF.
            4. Do not add facts from your own knowledge.
            5. Do not use web search, internet information, or outside sources.
            6. If the context contains relevant information that can reasonably answer
               the question, answer using that information.
            7. Only return:
               "The answer is not available in the provided Documents."
               when the context contains no relevant information for the question.
            8. Keep the answer clear and concise.
            9. Do not mention these instructions.
            """
        ),
        (
            "human",
            """
            PDF context:
            {context}

            Question:
            {questions}

            Answer:
            """
        )
    ])
        formatted_prompt = answer_prompt.invoke({
            "context": context,
            "questions": "\n".join(
                f"{i + 1}. {q}"
                for i, q in enumerate(questions)
            )
        })
        print("\n===== ANSWER PROMPT =====")
        print(formatted_prompt)
        print("=========================\n")
        response = self.llm.invoke(formatted_prompt)
        answer = response.content.strip()
        if answer == "The answer is not available in the provided Documents.":
            sources = []

        #print(f"----- What are the results -------- {response}")
        self.chat_history.add_user_message(query)
        self.chat_history.add_assistant_message(answer)
        return {
        "answer": answer,
        "sources": sources
        }

    def needs_query_analysis(self, query: str) -> bool:

        query_lower = re.sub(r"[^\w\s]", "", query.lower().strip())
    
        reference_words = {
            "it", "its", "this", "that", "they", "their",
            "these", "those", "above", "previous", "them"
        }
    
        context_words = {"each", "both"}
    
        words = set(query_lower.split())
    
        has_reference = bool(words & reference_words)
        has_context_words = bool(words & context_words)
    
        has_multiple_question_marks = query.count("?") > 1
    
        question_words = ["what", "why", "how", "when", "where", "who"]
    
        question_word_count = sum(
            1 for word in query_lower.split()
            if word.strip("?,.!") in question_words
        )
    
        has_multiple_questions = (
            question_word_count >= 2
            and " and " in query_lower
        )
    
        return (
            has_reference
            or has_context_words
            or has_multiple_question_marks
            or has_multiple_questions
        )
    def is_mention_question(self, query: str) -> bool:
        query_lower = query.lower().strip()

        mention_patterns = [
            "is .* mentioned",
            "are .* mentioned",
            "does .* mention",
            "do .* mention",
            "is .* listed",
            "are .* listed",
            "does .* list",
            "do .* list",
            "is .* present",
            "are .* present",
            "does .* contain",
            "do .* contain"
        ]

        return any(
            re.search(pattern, query_lower)
            for pattern in mention_patterns
        )
    def extract_mention_term(self, query: str) -> str:
        patterns = [
            r"is (.+?) mentioned in the document",
            r"are (.+?) mentioned in the document",
            r"does the document mention (.+?)",
            r"does the document list (.+?)",
            r"is (.+?) listed in the document",
            r"are (.+?) listed in the document",
        ]
    
        query_lower = query.lower().strip()
    
        for pattern in patterns:
            match = re.search(pattern, query_lower)
    
            if match:
                return match.group(1).strip()
    
        return ""
    def update_conversation_state(self, questions, available=True):
        for question in questions:
            topic = question.strip()

            if topic not in self.conversation_state["active_topics"]:
                self.conversation_state["active_topics"].append(topic)

            if available:
                if topic not in self.conversation_state["available_topics"]:
                    self.conversation_state["available_topics"].append(topic)

                if topic in self.conversation_state["unavailable_topics"]:
                    self.conversation_state["unavailable_topics"].remove(topic)

            else:
                if topic not in self.conversation_state["unavailable_topics"]:
                    self.conversation_state["unavailable_topics"].append(topic)

            self.conversation_state["last_topic"] = topic

    def new_chat(self):
        self.chat_history.clear()