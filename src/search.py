from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
load_dotenv()

class RAGSearch:
    def __init__(self, Vectorstore, chat_history, llm_model: str = "openai/gpt-oss-20b"):
        self.vectorstore = Vectorstore
        self.chat_history = chat_history
         # Load reranker 
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        GROQ_API_KEY  = os.getenv("GROQ_API_KEY")    
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY , model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def Search_and_summarize(self, query: str, top_k =3):
        #reteriver the context
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

            9. Return ONLY the rewritten question.

            10. Do not add explanations, labels, or quotation marks.
            """
        ),
        (
            "human",
            """
            Previous conversation:
            {chat_history}

            Current question:
            {query}

            Standalone question:
            """
        )
        ])
        
        formatted_prompt = rewrite_prompt.invoke({
                    "chat_history": self.chat_history.get_formatted_history(),
                    "query": query
                })
        
        response = self.llm.invoke(formatted_prompt)
        rewritten_query = response.content.strip()
        if not rewritten_query:
            rewritten_query = query

        print(f"[DEBUG] Original query: {query}")
        print(f"[DEBUG] Rewritten query: {rewritten_query}")

        results = self.vectorstore.query(
            rewritten_query,
            top_k=top_k
        )
        results = rerank(self, rewritten_query, results, top_k)
        print(f"[DEBUG] Retrieved chunk count: {len(results)}")
        
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        sources = [
        {
            "page": r["metadata"].get("page"),
            "file_name": r["metadata"].get("file_name")
        }
        for r in results
        if r["metadata"]
        ]
        context = "\n\n".join(texts)
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
            {rewritten_query}

            Answer:
            """
        )
        ])
        formatted_prompt = answer_prompt.invoke({
            "context": context,
            "rewritten_query": rewritten_query
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


def rerank(self, query, results, top_k=5):
    candidate_k = max(top_k * 3, 10)
    pairs = [
        (query, result["metadata"]["text"])
        for result in results
    ]

    scores = self.reranker.predict(pairs)

    ranked = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True
    )

    return [result for score, result in ranked[:top_k]]

def new_chat(self):
    self.chat_history.clear()