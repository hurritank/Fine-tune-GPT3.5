from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from utils import PDF_PATH, ENV_PATH, get_pdf_text
from typing import List
import os


# Load keys from environment
_ = load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]


def get_text_chunk(text: str) -> List:
    """
    Split document content to chunk
    """

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunk = text_splitter.split_text(text)

    return chunk


def get_vectorstore(text_chunks: List) -> FAISS:
    """
    Embedding and store text to vector database
    """
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectordb


def get_conversation_chain(vectordb: FAISS) -> BaseConversationalRetrievalChain:
    """
    Define conversation chain
    """
    llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0.6)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )

    return conversation_chain


pdf_text = get_pdf_text(PDF_PATH)

text_chunk = get_text_chunk(pdf_text)

vectorstore = get_vectorstore(text_chunk)

qa_chain = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    context = qa_chain({"question": "What is the purpose of generative agents?"})
    print(context)
    print("ANSWER")
    print(context["answer"])
