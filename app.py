# streamlit_app.py

import streamlit as st
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from operator import itemgetter

# API keys and configuration
google_api_key = "AIzaSyDrnMeMaCTgKPtyCVpkpCeIJ0OYSO7N32I"
qdrant_url = "https://5fddb2e4-0373-4695-b494-92f694bbe2a1.us-east4-0.gcp.cloud.qdrant.io"
qdrant_key = "F04KGqdN8yjWfSZ5kWai9TJc66WtoLWtOVgoKL4xKAuEPTkF_-MXrw"
collection_name = "Nowandthen"
embed_model_name = 'BAAI/bge-small-en-v1.5'  # The embedding model used

# Initialize Hugging Face embeddings
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

# Initialize Qdrant and Google PaLM (Gemini) APIs
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
llm = ChatGoogleGenerativeAI(api_key=google_api_key)

# Initialize Qdrant vector store and retriever
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding_model=embed_model
)

retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define helper function for formatting context documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define main question-answering function
def answer_query(query, history):
    # Retrieve relevant documents from Qdrant
    retrieved_docs = retriever.get_relevant_documents(query)
    context = format_docs(retrieved_docs)

    # Prompt template for the LLM with context
    prompt_str = """
    Answer the user's question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_str)
    response_chain = {
        "question": query,
        "context": context
    }

    # Execute response chain
    response = llm(response_chain)
    return response.content

# Streamlit Interface
st.title("Document Q&A with Qdrant and Google Gemini")
st.write("Ask a question based on the knowledge base:")

# Chat history to provide continuity
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Your question:", "")
submit_button = st.button("Submit")

if submit_button and user_input:
    # Get answer from Qdrant-Gemini system
    response = answer_query(user_input, st.session_state.history)
    
    # Display answer
    st.write(f"**Answer:** {response}")

    # Add to history
    st.session_state.history.append({"user_question": user_input, "ai_response": response})
    
    # Display chat history
    st.write("### Conversation History:")
    for entry in st.session_state.history:
        st.write(f"**User:** {entry['user_question']}")
        st.write(f"**AI:** {entry['ai_response']}")
