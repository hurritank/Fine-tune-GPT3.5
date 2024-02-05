import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from prompts import ROLE_PROMPT
from utils import ENV_PATH
from rag_langchain import qa_chain


app = FastAPI()

# Load keys from environment
_ = load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FINE_TUNED_MODEL_NAME = os.environ["FINE_TUNED_MODEL_NAME"]


class MessageItem(BaseModel):
    message: str


@app.get("/")
def home():
    """
    Test API
    """
    return "API IS RUNNING!"


@app.post("/chatgpt-chat")
def chatgpt_api(item: MessageItem) -> str:
    # Settings OpenAI Key
    openai.api_key = OPENAI_API_KEY
    message = item.message
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    messages = [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": message},
    ]
    completion = openai.ChatCompletion.create(
        model=FINE_TUNED_MODEL_NAME,
        messages=messages
    )

    return completion.choices[0].message["content"]


@app.post("/rag-chat")
def rag_api(item: MessageItem) -> str:
    # Settings OpenAI Key
    openai.api_key = OPENAI_API_KEY
    message = item.message
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    prompt = {"question": message}
    context = qa_chain(prompt)['answer']

    return context






