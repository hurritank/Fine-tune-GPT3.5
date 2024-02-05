import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai

PDF_PATH = "data/generative_agent.pdf"
TXT_PATH = "data/generative_agents_content.txt"
CSV_PATH = "data/generative_agent_data.csv"
FINE_TUNE_FILE_PATH = "data/fine_tune_data.csv"
TRAIN_PATH = "data/train.jsonl"
VALID_PATH = "data/valid.jsonl"
ENV_PATH = ".env"

# Load keys from environment
_ = load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]


# def get_and_create_dataset(pdf_path: str, text_path: str) -> str:
#     """
#     Read pdf file then write content to txt file
#     """
#     # Load PDF
#     with open(pdf_path, "rb") as f:
#         pdf = pdftotext.PDF(f)
#
#     content = "\n\n".join(pdf)
#
#     # Write to txt file
#     with open(text_path, 'w') as f:
#         f.write(content)
#
#     return text_path


def get_pdf_text(pdf_path: str) -> str:

    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def call_openai(default_prompt: str, context: str) -> str:
    """
    Call Open AI api
    """
    # Settings OpenAI Key
    openai.api_key = OPENAI_API_KEY
    # Adding prompt
    prompt = default_prompt + context
    # Setting prompt
    final_prompt = [{'role': 'system', 'content': prompt}]
    # Call api
    response = openai.ChatCompletion.create(model=OPENAI_MODEL_NAME,
                                            messages=final_prompt,
                                            temperature=0)
    return response.choices[0].message["content"]
