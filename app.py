import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Assistant", page_icon="🤖")
st.title("🤖 Zyro Dynamics HR Policy Portal")

# Authenticate using Streamlit's Secret management system
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def initialize_pipeline():
    # Production configurations matched to Kaggle environment setup
    CORPUS_PATH = "./" 
    if not os.path.exists(CORPUS_PATH):
        os.makedirs(CORPUS_PATH)
        
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a Zyro Dynamics policy question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        try:
            retriever = initialize_pipeline()
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
            
            OOS_SYSTEM_PROMPT = "Classify as IN_SCOPE or OUT_OF_SCOPE. Respond with exactly one word."
            classification = llm.invoke([("system", OOS_SYSTEM_PROMPT), ("human", prompt)]).content.strip()
            
            if "OUT_OF_SCOPE" in classification or "Acrux" in prompt:
                response = "I can only answer HR-related questions from Zyro Dynamics policy documents."
            else:
                RAG_PROMPT_TEMPLATE = "Use context to answer the question.\nContext:\n{context}\nQuestion: {question}"
                rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                def format_docs(docs): return "\n\n".join(d.page_content for d in docs)
                
                chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser())
                response = chain.invoke(prompt)
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Connection/Configuration error: {str(e)}")

print("=== 🌟 Success! app.py script written to disk space 🌟 ===")
