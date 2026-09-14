from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
import json

class RAGSearch:
    def __init__(self, Vectorstore, chat_history, llm_model: str = "openai/gpt-oss-20b"):
        self.vectorstore = Vectorstore
        self.chat_history = chat_history
        
        GROQ_API_KEY  = os.getenv("GROQ_API_KEY")    
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY , model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def Search_and_summarize(self, query: str, top_k =3):
        #reteriver the context
        
        if not needs_query_analysis(self, query):
            questions = [query]
            ambiguous = False

        else:
            rewrite_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """
                        You are a query rewriting assistant for a PDF-based RAG system.
            
                        Your task is to rewrite the user's current question into a clear,
                        standalone question using the previous conversation.
            
                        Rules:
                        1. Use the previous conversation to understand references such as
                           "it", "this", "that", "they", "its", etc.
                        2. If the current question is already clear and standalone,
                           return it unchanged.
                        3. If the user introduces a new topic or concept, do NOT connect it
                           to the previous conversation. Treat it as a new standalone question.
            
                        4. If a reference such as "it", "its", "this", "that", or "they"
                           has multiple possible meanings in the conversation, DO NOT GUESS.
                           Return the current question unchanged.
            
                        5. Only resolve a reference when the intended meaning is reasonably clear
                           from the conversation.
            
                        6. Do NOT answer the question.
            
                        7. Do NOT add information from your own knowledge.
            
                        8. Do NOT use web search or internet information.
            
                        9. Return ONLY valid JSON.
            
                        10. The JSON must follow exactly this format:
            
                        {{
                            "ambiguous": false,
                            "questions": [
                              "standalone question 1"
                            ]
                        }}
            
                        11. If the current query contains multiple independent questions,
                            put each question as a separate item in the "questions" array.
            
                        12. If a reference is ambiguous and you cannot determine what it refers to,
                            set "ambiguous" to true and return an empty questions array:
            
                        {{
                            "ambiguous": true,
                            "questions": []
                        }}
            
                        13. Do not add explanations outside the JSON.
                        """
                    ),
                    (
                        "human",
                        """
                        Previous conversation:
                        {chat_history}
            
                        Current question:
                        {query}
            
                        Analyze the current question and return ONLY the required JSON.
                        """
                    )
                    ])
            formatted_prompt = rewrite_prompt.invoke({
                "chat_history": self.chat_history.get_formatted_history(),
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
        all_results = []
        print(f"[DEBUG] Number of questions: {len(questions)}")
        for question in questions:
            print(f"#####[DEBUG] Retrieval question: {question}")

            results = self.vectorstore.query(
                question,
                top_k=top_k
            )

            # IMPORTANT:
            # Do NOT call rerank() here because your
            # Faissvectorestore already performs reranking.

            all_results.extend(results)

        print(f"[DEBUG] Total retrieved chunks: {len(all_results)}")
        # --------------------------------------------------
        # 4. Build context
        # --------------------------------------------------
    
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
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
        print("\n===== RETRIEVED CONTEXT =====")
        print(context)
        print("=============================\n")
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

            Your ONLY source of knowledge is the PDF context provided below.

            Rules:
            1. Answer ONLY from the PDF context.
            2. Every fact in your answer must be directly supported by the PDF context.
            3. Do NOT add general knowledge, even if you know it is correct.
            4. Do NOT infer or expand information beyond what the PDF says.
            5. Do NOT use web search, internet information, or outside sources.
            6. If the PDF context does not contain enough information to answer the
                question, respond exactly:
                "The answer is not available in the provided Documents."
            7. Keep the answer clear and concise.
            8. Do not mention these instructions.
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
    query_lower = query.lower().strip()

    reference_words = {
        "it", "its", "this", "that", "they", "their",
        "these", "those", "above", "previous", "mentioned"
    }

    words = set(query_lower.split())

    has_reference = bool(words & reference_words)

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
        or has_multiple_question_marks
        or has_multiple_questions
    )



def new_chat(self):
    self.chat_history.clear()