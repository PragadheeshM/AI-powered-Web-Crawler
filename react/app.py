from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
from typing import List, Optional, Any
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from fpdf import FPDF
import groq
import base64


os.environ["USER_AGENT"] = "MyLangChainApp/1.0"
# Load .env variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global store (or use Redis/DB in production)
vector_store = None

# --- Groq LLM Wrapper ---
class GroqChatLLM(BaseChatModel):
    model_name: str = Field(default="llama3-8b-8192")
    temperature: float = 0.7
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    client: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = groq.Client(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = []
        for msg in messages:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            formatted_messages.append({
                "role": role,
                "content": str(msg.content)
            })

        chat_completion = self.client.chat.completions.create(
            messages=formatted_messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=1000,
            stop=stop
        )

        content = chat_completion.choices[0].message.content
        generation = ChatGeneration(message=AIMessage(content=content), text=content)
        return ChatResult(generations=[generation])

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return self._generate(messages, stop=stop, **kwargs)

# --- Vector Store Setup ---
def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        if not document:
            raise ValueError("No content found on the page.")

        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_documents(document)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(chunks, embedding=embeddings)
    except Exception as e:
        print(f"[VECTORSTORE] Error loading content from {url}: {e}")
        raise
def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        if not document:
            raise ValueError("No content found on the page.")

        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_documents(document)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(chunks, embedding=embeddings)
    except Exception as e:
        print(f"[VECTORSTORE] Error loading content from {url}: {e}")
        raise


# --- RAG Pipeline ---
def get_context_retriever_chain(vector_store):
    llm = GroqChatLLM()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query to get relevant info from the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = GroqChatLLM()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions using this context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, chain)

# --- Chat Handler ---
def your_chat_function(user_input, chat_history, document_url=None):
    global vector_store
    if vector_store is None and document_url:
        vector_store = get_vectorstore_from_url(document_url)

    if vector_store is None:
        raise ValueError("Vector store not initialized. Please provide a valid document URL first.")

    retriever_chain = get_context_retriever_chain(vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)

    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    if isinstance(response, dict) and "answer" in response:
        return response["answer"]
    else:
        return str(response)



# --- Flask API Routes ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    chat_history = data.get('chat_history', [])
    document_url = data.get('document_url', None) 
    try:
        result = your_chat_function(user_message, chat_history)
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pdf', methods=['POST'])
def generate_pdf():
    content = request.get_json().get("content", "No content provided")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf_path = "output.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)


@app.route('/api/scrape', methods=['POST'])
def scrape_website():
    global vector_store
    try:
        data = request.get_json()
        url = data.get("url")
        print(f"[SCRAPE] URL received: {url}")

        if not url:
            return jsonify({"error": "URL is required"}), 400

        vector_store = get_vectorstore_from_url(url)
        print("[SCRAPE] Vector store created successfully")

        return jsonify({"message": "Website content scraped and vectorized successfully."})
    except Exception as e:
        print(f"[SCRAPE] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)