from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history, mention the required details in the question itself, also mention the NCT ID if present in the previous trial.\
    Just return the standalone question, Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.

Chat History:\"""
{chat_history}
\"""
Question: \"""
{question}
\"""

 """

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
Give the answer in as much detailed way as possible.\

Context :\"""
{context}
\"""
Question : \"""
{question}
\"""

 """

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(contextualize_q_system_prompt)

QA_PROMPT = ChatPromptTemplate.from_template(qa_system_prompt)