from typing import Any, Callable, Dict, Optional
from langchain.schema import AIMessage, HumanMessage
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from template import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from operator import itemgetter
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
import numpy as np
import json
from pathlib import Path
from openai import OpenAI
import openai

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

class ModelWrapper:

    def __init__(self, model):
        self.llm = model
        self.api_key = st.secrets["openai-key"]
        self.load_model()
    
    def load_model(self):
        self.main_model = ChatOpenAI(model_name=self.llm, temperature=0.1, api_key = self.api_key, streaming = True)
        self.rephrase_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1, api_key = self.api_key)

    def conversational_chain(self, vectorstore):

        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)
        
        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | self.rephrase_model
            | StrOutputParser()
        )

        _context = {
            "context": itemgetter("standalone_question")
            | vectorstore.as_retriever()
            | _combine_documents,
            "question": lambda x: x["standalone_question"],
        }

        conversational_qa_chain = _inputs | _context | QA_PROMPT | self.main_model

        return conversational_qa_chain

    def retrieval_chain(self, vectorstore):
    
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history.\
        Include all necessary details within the question itself.\
        If an NCT ID is mentioned in the chat history and is relevant to the question, ensure it is included in the reformulated question\
        Just return the standalone question, Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If user asks for suggestions/options at least provide 3 options\
        
        Please provide a concise answer. \

        {context}"""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
                [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ]
        )

        retriever = vectorstore.as_retriever(k=5)
        history_aware_retriever = create_history_aware_retriever(self.rephrase_model, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(self.main_model, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return retrieval_chain

class Identify:

    def __init__ (self, chat_history, input):
        self.client = OpenAI(api_key = st.secrets["openai-key"])
        self.chat_history = chat_history
        prompt = """Given the chat history and the latest user question,\
        you need to determine whether the question pertains to clinical trials, medical studies, any aspect of the medical field, or greetings messages such as Hi, Hello. \
        If the question is related to any of these topics, respond with "Yes." \
        If the question does not relate to these topics or includes a mixture of greetings and unrelated subjects, respond with "No."\
        It is important to note that you should not provide an answer to the user question itself;\
        DO NOT ANSWER THE QUESTION, just simply respond with "Yes" or "No" based on the relevance criteria outlined."""
        self.input = input + "\n" + prompt
        
    def identify_chain(self):

        system_prompt = """Given the chat history and the latest user question,\
        you need to determine whether the question pertains to clinical trials, medical studies, any aspect of the medical field, or greetings messages such as Hi, Hello. \
        If the question is related to any of these topics, respond with "Yes." \
        If the question does not relate to these topics or includes a mixture of greetings and unrelated subjects, respond with "No."\
        It is important to note that you should not provide an answer to the user question itself;\
        DO NOT ANSWER THE QUESTION, just simply respond with "Yes" or "No" based on the relevance criteria outlined."""
        
        main_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1, api_key = st.secrets["openai-key"])
        qa_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )

        qa_chain = qa_prompt | main_model
        ans = qa_chain.invoke({"input": self.input, "chat_history": self.chat_history})
        self.result = ans.content
        print("This is result:", ans)
        print("\n")
        print("debug:",self.result)
        print("\n")
        
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def determine_answer(self):
    
        embeddings = self.client.embeddings.create(input=["Yes", "No", self.result], model="text-embedding-ada-002")
        vector1 = embeddings.data[0].embedding
        vector2 = embeddings.data[1].embedding
        vector3 = embeddings.data[2].embedding

        score_yes = self.cosine_similarity(vector1, vector3)
        score_no = self.cosine_similarity( vector2, vector3)
        return True if score_yes > score_no else False

def question_relatable(chat_history, input):
    obj = Identify(chat_history, input)
    obj.identify_chain()
    result = obj.determine_answer()

    return result

def process_messages(messages_list):
    langchain_messages = []
    for messages_dict in messages_list:
        role = messages_dict.get('role')
        content = messages_dict.get("content")

        if role == "user":
            langchain_messages.append(HumanMessage(content = content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content = content))

    return langchain_messages

def load_chain(model_name="gpt-3.5-turbo", chain = "retrieval_chain"):

    api_key = st.secrets["openai-key"]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local("database/vectorDB/UCDB1", embeddings, allow_dangerous_deserialization=True)

    model = ModelWrapper(model = model_name)
    return model.retrieval_chain(vectorstore), vectorstore if chain == "retrieval_chain" else model.conversational_chain(vectorstore)

def find_pattern(matches):
    
    data = json.loads(Path("./data.json").read_text())
    print("hello")
    docs = []
    for protocol in data:
        if protocol["NCT_ID"] in matches:
            text = protocol["NCT_ID"] + " "
            if "TITLE" in protocol:
                text += "Title: " + protocol["TITLE"] + " "
            text += "Short Title: " + protocol["SHORT_TITLE"] + " "
            text += "Sponsor: " + protocol["SPONSOR"] + " "
            text += "Detailed Eligibility: " + protocol["DETAILED_ELIGIBILITY"] + " "
            if "DESCRIPTION" in protocol:
                text += "Description: " + protocol["DESCRIPTION"] + " "
            text += "Summary: " + protocol["SUMMARY"] + " "
            text += "Status: " + protocol["STATUS"] + " "
            if  "PRIMARY_OUTCOMES" in protocol:
                text += "Outcome Description: " + protocol["PRIMARY_OUTCOMES"] + " "
            if "SECONDARY_OUTCOMES" in protocol:
                text += "Outcome Measure: " + protocol["SECONDARY_OUTCOMES"] + " "
            if "OTHER_OUTCOMES" in protocol:
                text += "Outcome Timeframe: " + protocol["OTHER_OUTCOMES"] + " "
            text += "Age Description: " + protocol["AGE_DESCRIPTION"] + " "
            if "CONDITIONS" in protocol:
                text += "Conditions: " + str(protocol["CONDITIONS"])+ " "
            if "OVERALL_OFFICIALS" in protocol:
                text += "Overall Officials: " + protocol["OVERALL_OFFICIALS"]+ " "
            if "LOCATIONS" in protocol:
                text += "Locations: " + protocol["LOCATIONS"]+ " "
            
            docs.append(Document(page_content=text, metadata={"source": protocol["NCT_ID"]}))

    print("Hello2")
    return docs if len(docs)>1 else docs



def load_new_history_chain( chat_history, input, docs):

    api_key = st.secrets["openai-key"]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local("database/vectorDB/UCDB1", embeddings, allow_dangerous_deserialization=True)
    model = ChatOpenAI(model_name='gpt-4o', temperature=0.1, api_key = api_key)
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    retriever = vectorstore.as_retriever(k=3)
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    result = history_aware_retriever.invoke({"chat_history": chat_history,"input":input})
    result.extend(docs)
    source_knowledge = "\n".join([x.page_content for x in result])
    # print(source_knowledge)
    return source_knowledge 

def load_new_qa_chain():

    api_key = st.secrets["openai-key"]
    model = ChatOpenAI(model_name='gpt-4o', temperature=0.1, api_key = api_key)

    qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        Please provide a concise answer. \

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
    )

    qa_chain =  qa_prompt | model | StrOutputParser()

    return qa_chain