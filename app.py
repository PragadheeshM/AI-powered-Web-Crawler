import streamlit as st
from dotenv import load_dotenv
import os
import groq
from typing import ClassVar, List, Optional, Any, Dict, Type, Union

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field
import streamlit.components.v1 as components
from fpdf import FPDF
import base64

load_dotenv()

class GroqChatLLM(BaseChatModel):
    model_name: str = Field(default="llama3-8b-8192")
    temperature: float = 0.7
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    client: Any = None  # Add client field
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        print(f"Using model: {self.model_name}")
        print(f"API key: {self.api_key[:5]}...{self.api_key[-5:] if self.api_key else ''}")
        self.client = groq.Client(api_key=self.api_key) if self.api_key else None

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
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        
        # Convert LangChain messages to Groq format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                role = "assistant"
            else:  # Default to user for HumanMessage and other types
                role = "user"
                
            formatted_messages.append({
                "role": role,
                "content": str(msg.content)
            })
        
        try:
            # Make the API call using Groq client
            chat_completion = self.client.chat.completions.create(
                messages=formatted_messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=1000,
                stop=stop
            )
            
            # Extract the response content
            if chat_completion.choices and chat_completion.choices[0].message:
                content = chat_completion.choices[0].message.content
                # Create a ChatGeneration object
                generation = ChatGeneration(
                    message=AIMessage(content=content),
                    text=content,
                )
                # Return a proper ChatResult object
                return ChatResult(generations=[generation])
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"Error details: {error_msg}")
            raise Exception(error_msg)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        return self._generate(messages, stop=stop, **kwargs)

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(document_chunks, embedding=embeddings)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = GroqChatLLM()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = GroqChatLLM()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)

    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response["answer"]

# Streamlit App Setup
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

def generate_chat_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for msg in chat_history:
        sender = "User" if isinstance(msg, HumanMessage) else "Bot"
        content = msg.content.replace("\n", "\n")
        pdf.multi_cell(0, 10, f"{sender}: {content}\n", align='L')
    
    return pdf.output(dest="S").encode("latin1")

def render_pdf_download_button(pdf_bytes, filename="chat_history.pdf"):
    st.download_button(
        label="📥 Download Chat as PDF",  # Button label
        data=pdf_bytes,                  # PDF bytes
        file_name=filename,             # File name on download
        mime="application/pdf"          # MIME type
    )

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]   

    if "vector_store" not in st.session_state:
        with st.spinner("Indexing website..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    
    # Add a download button for PDF
    pdf_bytes = generate_chat_pdf(st.session_state.chat_history)
    st.markdown(render_pdf_download_button(pdf_bytes), unsafe_allow_html=True)


    # Chat rendering
    for idx, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                unique_id = f"ai_message_{idx}"
                components.html(f"""
                    <div style="position: relative; background-color: #f1f1f1; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                        <pre id="{unique_id}" style="white-space: pre-wrap; word-wrap: break-word; font-family: inherit;">{message.content}</pre>
                        <button 
                            onclick="navigator.clipboard.writeText(document.getElementById('{unique_id}').innerText)" 
                            style="position: absolute; top: 5px; right: 5px; background-color: #e5e7eb; border: none; border-radius: 5px; padding: 5px; cursor: pointer;"
                        >📋 Copy</button>
                    </div>
                """, height=100 + 20 * message.content.count('\n'))
        else:
            with st.chat_message("Human"):
                st.write(message.content)


else:
    st.info("Please enter a website URL in the sidebar.")



