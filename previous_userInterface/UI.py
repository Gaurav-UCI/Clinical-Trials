# Import necessary libraries
import databutton as db
import streamlit as st
import openai
import time 
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOpenAI
from preprocess import custom_search
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import os
import re 

st.set_page_config(layout="wide")

st.title("Clinical Trials RAG Chatbot")

with open(r"./openai-key/openai_key.txt", 'r') as file:
    api_key = file.read().strip()
    
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

# st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

prompt_template = """

        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer creative.
        
        Please take into account the previous messages as well.
        
        Make sure to citation for the answer from metadata.
            
        Reply to greetings messages.
    """

if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": prompt_template}, {"role": "user", "content": "Hi! How are you?"}, {"role": "assistant", "content": "Hello, How can I help you?"}]

prompt = st.session_state.get("prompt")

question = st.chat_input("Ask anything")

for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
    

if question:
    

    faiss_path = r"./database/vectorDB/mainDB"
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")

   
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that "Apologies, I'm unable to provide a response to that inquiry at this moment. For further assistance, please feel free to reach out to us via phone at 714-456-7890 or visit our website at ucihealth.org. We'll be happy to help you there.". \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()

    ai_msg = rag_chain.invoke({"input": question, "chat_history": prompt})["answer"]

    if ai_msg is not None:
        botmsg.write(ai_msg)

    prompt.append({"role": "user", "content": question})
    prompt.append({"role": "assistant", "content": ai_msg})

    st.session_state["prompt"] = prompt

