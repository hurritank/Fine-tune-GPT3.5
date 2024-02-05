```

```

# Fine-tune-GPT3.5

Fine-tuning ChatGPT3.5

### Directory Structure

```commandline
llm-du
    |───src/                                  # Models directory
        |───data/                             # Store raw and processed dataset for fine-tune
            |───fine_tune_data.csv            # Processed dataset, only True data used to create train and validate set
            |───generative_agent.pdf          # Pdf file to extract content
            |───generative_agent_data.csv     # Generative dataset with True and False data
            |───generative_agents_content.txt # Raw content extract from pdf file
            |───train.jsonl                   # Train dataset use to train by openai
            |───valid.jsonl                   # Validation dataset use to validate by openai
        |───.env                              # Define enviroment variables
        |───api.py                            # Define API for chatgpt api and RAG api
        |───fine_tune_chatgpt.py              # Fine-tune chatGPT3.5 script
        |───prepare_dataset.py                # Prepare fine-tune chatGPT3.5 dataset script
        |───prompts.py                        # Define prompts for chatGPT
        |───rag_langchain.py                  # Define RAG for search context for question answer
        |───utils.py                          # Define parameters for project
        |───requirements.txt                  # Library requirements
    └───README.md                             # Project introductions
```

### Installation

Copy .env file and put in src folder

#### Run with Docker

docker compose up --build

#### Run manually:

- OS: Windows/Linux/MAC
- Python == 3.9

#### Install Dependencies:

> pip install -r requirements.txt

#### API request:

Run API:
Incase using OCR API, uncomment

> cd src
>
> uvicorn api:app --reload

Documents:
open url and add /docs to see document, e.g:

> http://127.0.0.1:8000/docs

##### API Endpoint

```
/chatgpt-chat # chat with fine-tuned ChatGPT3.5
/rag-chat     # chat with RAG
```
