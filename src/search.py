from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

class RAGSearch:
    def __init__(self, Vectorstore,chat_history, persist_dir: str="faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-20b"):
        self.vectorstore = Vectorstore
        self.chat_history = chat_history
        # Load or build vectorstore
        store_dir = getattr(self.vectorstore, "persist_dir", persist_dir)
        faiss_path = os.path.join(store_dir, "faiss.index")
        meta_path = os.path.join(store_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import Load_all_documents
            docs = Load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        GROQ_API_KEY  = os.getenv("GROQ_API_KEY")    
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY , model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def Search_and_summarize(self, query: str, top_k =3):
        #reteriver the context
        rewrite_prompt = f"""
        You are a query rewriting assistant for a PDF-based RAG system.

        Your task is to rewrite the user's current question into a clear,
        standalone question using the previous conversation.

        Rules:
        1. Use the previous conversation to understand references such as
           "it", "this", "that", "they", "its", etc.
        2. If the current question is already clear and standalone,
           return it unchanged.
        3. Do NOT answer the question.
        4. Do NOT add information from your own knowledge.
        5. Do NOT use web search or internet information.
        6. Return ONLY the rewritten question.
        7. Do not add explanations, labels, or quotation marks.

        Previous conversation:
        {self.chat_history.get_formatted_history()}

        Current question:
        {query}

        Standalone question:
        """
        
        rewritten_query = self.llm.invoke([rewrite_prompt]).content.strip()
        print(f"The rewritten Query by the LLM is '{rewritten_query}'")
        results = self.vectorstore.query(rewritten_query, top_k= top_k)
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
            return "No relevent context found to answer the question."
        ## Generate the answer uisng Groq LLM
        answer_prompt = f"""
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
           "The answer is not available in the provided PDF."
        7. Keep the answer clear and concise.
        8. Do not mention these instructions.
        
        PDF context:
        {context}
        
        Question:
        {rewritten_query}
        
        Answer:
        """
        response = self.llm.invoke([answer_prompt])
        #print(f"----- What are the results -------- {response}")
        self.chat_history.add_user_message(query)
        self.chat_history.add_assistant_message(response.content)
        return {
        "answer": response.content,
        "sources": sources
        }

    def new_chat(self):
        self.chat_history.clear()