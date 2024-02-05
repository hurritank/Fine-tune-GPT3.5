import os
import dotenv
from dotenv import load_dotenv
import openai
from typing import Dict, Optional
from utils import ENV_PATH, TRAIN_PATH, VALID_PATH

# Load keys from environment
_ = load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]


def fine_tune_chatgpt() -> Dict:

    # Upload training data
    train = openai.File.create(
        file=open(TRAIN_PATH, "rb"),
        purpose="fine-tune"
    )
    # Upload validate data
    valid = openai.File.create(
        file=open(VALID_PATH, "rb"),
        purpose="fine-tune"
    )
    # Create a fine-tune model
    response = openai.FineTuningJob.create(
        training_file=train["id"],
        validation_file=valid["id"],
        model=OPENAI_MODEL_NAME
    )
    print("Created fine-tune job")

    return response


def get_fine_tuned_model(response: Dict) -> Optional[str]:

    # Fine tune job ID
    fine_tune_job_id = response["id"]
    # Get state of fine-tune
    fine_tune_job = openai.FineTuningJob.retrieve(fine_tune_job_id)
    # If done
    if fine_tune_job["finished_at"]:
        # Monitoring loss
        events = openai.FineTuningJob.list_events(id=fine_tune_job_id, limit=10)["data"]
        events.reverse()
        for event in events:
            print(event["message"])

        return fine_tune_job["fine_tuned_model"]
    else:
        print("Please wait and try again")
        return None


if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    response = fine_tune_chatgpt()
    print(response)
    if response["id"]:
        os.environ["FINE_TUNE_JOB_ID"] = response["id"]
        dotenv.set_key(ENV_PATH, "FINE_TUNE_JOB_ID", os.environ["FINE_TUNE_JOB_ID"])
    model_id = get_fine_tuned_model(response)
    if model_id:
        print(model_id)
        os.environ["FINE_TUNED_MODEL_NAME"] = model_id
        dotenv.set_key(ENV_PATH, "FINE_TUNED_MODEL_NAME", os.environ["FINE_TUNED_MODEL_NAME"])







