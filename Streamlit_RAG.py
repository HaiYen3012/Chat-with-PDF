# Tạo giao diện người dùng Chatbot
import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
from llm import load_normal_chain
from langchain.memory import StreamlitChatMessageHistory

# Tạo thư mục tạm thời
TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data','tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'vector_store')

header = st.container()

def streamlit_ui():
    with st.sidebar:
        choice = option_menu('Table of contents', ['Home', 'Chat with document/RAG'])
    if choice == 'Home':
        RAG_HOME()

    elif choice == 'Chat with document/RAG':
        with header:
            st.title('Chat with document/RAG')
            st.write('Upload a document that you want to chat')
            source_docs = st.file_uploader(label="Upload a document", type=['pdf'],accept_multiple_files=True)
            if not source_docs:
                st.warning("Please upload a document")
            else:
                for source_doc in source_docs:
                    # Kiểm tra loại tệp tin
                    if source_doc.type == 'application/pdf':
                        st.success(f"{source_doc.name} is a valid PDF file.")
                    else:
                        st.error(f"{source_doc.name} is not a valid PDF file. Please upload a PDF file.")
                        continue

                    # Kiểm tra kích thước tệp tin
                    if source_doc.size > 0:
                        st.success(f"{source_doc.name} has a size of {source_doc.size} bytes.")
                    else:
                        st.error(f"{source_doc.name} is empty. Please upload a non-empty PDF file.")
                        continue
                    
                query = st.chat_input()
                RAG(source_docs, query)

def RAG(docs,query):
    # Load the document
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())

    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    # Vector and embeddings
    DB_FAISS_PATH = 'vectorestore_lmstudio/faiss'
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device':'cpu'})
    db = FAISS.from_documents(text,embeddings)
    db.save_local(DB_FAISS_PATH)

    # Setup LLM. Fetch base_url from LM Studio
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Build a conversational chain
    qa_chain=ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True
    )

    chat_history=[]
    result = qa_chain({'question':query, 'chat_history':chat_history})
    st.write(result['answer'])
    chat_history.append((query, result['answer']))


def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def RAG_HOME():
    st.title("Multimodal Local Chat App")
    chat_container = st.container()

    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ""

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)

    send_button = st.button("Send", key="send_button")

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
    
            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                llm_response = llm_chain.run(st.session_state.user_question)
                st.session_state.user_question = ""

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History: ")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)
    
streamlit_ui()