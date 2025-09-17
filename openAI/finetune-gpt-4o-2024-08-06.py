import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

import openai
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from a .env file if present
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='fine_tune.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to orchestrate the fine-tuning and evaluation process.
    """
    # Configuration settings
    config = {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'training_file_path': './datasets/train_data.jsonl',
        'test_file_path': './datasets/test_data.jsonl',
        'model': 'gpt-4o-2024-08-06',
        'n_epochs': 4,
        'batch_size': None,  # Let OpenAI choose the appropriate batch size
        'learning_rate_multiplier': None,
        'suffix': 'svd_finetune',
    }

    # Validate API key
    if not config['api_key']:
        logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=config['api_key'])

    # Upload training file
    training_file_id = upload_file(client, config['training_file_path'], 'Training')
    if not training_file_id:
        logger.error("Failed to upload training file.")
        print("Error: Failed to upload training file.")
        return

    # Create fine-tuning job
    fine_tune_job_id = create_fine_tune_job(
        client,
        training_file_id,
        config['model'],
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        learning_rate_multiplier=config['learning_rate_multiplier'],
        suffix=config['suffix']
    )

    if not fine_tune_job_id:
        logger.error("Failed to create fine-tuning job.")
        print("Error: Failed to create fine-tuning job.")
        return

    # Monitor fine-tuning job
    fine_tuned_model = monitor_fine_tune_job(client, fine_tune_job_id)

    if fine_tuned_model:
        print(f"Fine-tuned model is ready: {fine_tuned_model}")
        logger.info(f"Fine-tuned model is ready: {fine_tuned_model}")
        # Evaluate the fine-tuned model on the test dataset
        evaluate_model(client, fine_tuned_model, config['test_file_path'])
    else:
        logger.error("Fine-tuning job did not complete successfully.")
        print("Error: Fine-tuning job did not complete successfully.")

def upload_file(client: OpenAI, file_path: str, file_type: str) -> Optional[str]:
    """
    Uploads a file to OpenAI for fine-tuning.

    Args:
        client (OpenAI): The OpenAI client instance.
        file_path (str): Path to the JSONL file.
        file_type (str): Type of the file ('Training').

    Returns:
        Optional[str]: The file ID if upload is successful, None otherwise.
    """
    try:
        logger.info(f"Uploading {file_type.lower()} file: {file_path}")
        print(f"Uploading {file_type.lower()} file: {file_path}")

        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"{file_type} file not found at path: {file_path}")
            print(f"Error: {file_type} file not found at path: {file_path}")
            return None

        # Upload file
        response = client.files.create(
            file=Path(file_path),
            purpose='fine-tune',
        )

        file_id = response.id
        logger.info(f"{file_type} file uploaded successfully. File ID: {file_id}")
        print(f"{file_type} file uploaded successfully. File ID: {file_id}")
        return file_id
    except openai.APIError as e:
        logger.error(f"OpenAI API error during {file_type.lower()} file upload: {e}")
        print(f"Error: OpenAI API error during {file_type.lower()} file upload: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during {file_type.lower()} file upload: {e}")
        print(f"Error: Unexpected error during {file_type.lower()} file upload: {e}")
        return None

def create_fine_tune_job(
    client: OpenAI,
    training_file_id: str,
    model: str,
    n_epochs: int = 2,
    batch_size: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    suffix: Optional[str] = None
) -> Optional[str]:
    """
    Creates a fine-tuning job.

    Args:
        client (OpenAI): The OpenAI client instance.
        training_file_id (str): The ID of the uploaded training file.
        model (str): The base model to fine-tune.
        n_epochs (int): Number of epochs to train the model.
        batch_size (Optional[int]): Batch size to use during training.
        learning_rate_multiplier (Optional[float]): Learning rate multiplier.
        suffix (Optional[str]): Suffix for the fine-tuned model name.

    Returns:
        Optional[str]: The fine-tuning job ID if successful, None otherwise.
    """
    try:
        logger.info(f"Creating fine-tuning job for model: {model}")
        print(f"Creating fine-tuning job for model: {model}")

        # Prepare hyperparameters
        hyperparams = {}
        if n_epochs is not None:
            hyperparams['n_epochs'] = n_epochs
        if batch_size is not None:
            hyperparams['batch_size'] = batch_size
        if learning_rate_multiplier is not None:
            hyperparams['learning_rate_multiplier'] = learning_rate_multiplier

        # Create fine-tuning job
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            hyperparameters=hyperparams,
            suffix=suffix
        )
        job_id = response.id
        logger.info(f"Fine-tuning job created successfully. Job ID: {job_id}")
        print(f"Fine-tuning job created successfully. Job ID: {job_id}")
        return job_id
    except openai.APIError as e:
        logger.error(f"OpenAI API error during fine-tuning job creation: {e}")
        print(f"Error: OpenAI API error during fine-tuning job creation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during fine-tuning job creation: {e}")
        print(f"Error: Unexpected error during fine-tuning job creation: {e}")
        return None

def monitor_fine_tune_job(client: OpenAI, job_id: str) -> Optional[str]:
    """
    Monitors the fine-tuning job until completion, retrieving and logging fine-tuning events.

    Args:
        client (OpenAI): The OpenAI client instance.
        job_id (str): The ID of the fine-tuning job.

    Returns:
        Optional[str]: The fine-tuned model name if successful, None otherwise.
    """
    logger.info(f"Monitoring fine-tuning job: {job_id}")
    print(f"Monitoring fine-tuning job: {job_id}")
    status = ''
    fine_tuned_model = None
    try:
        while True:
            # Retrieve job status
            response = client.fine_tuning.jobs.retrieve(job_id)
            new_status = response.status
            if new_status != status:
                status = new_status
                logger.info(f"Job status: {status}")
                print(f"Job status: {status}")

            # Retrieve and log fine-tuning events
            events = client.fine_tuning.jobs.list_events(job_id)
            for event in events.data:
                message = event.message
                level = event.level
                logger.info(f"Event [{level}]: {message}")

            # Check for completion
            if status == 'succeeded':
                fine_tuned_model = response.fine_tuned_model
                logger.info(f"Fine-tuning job completed successfully. Fine-tuned model: {fine_tuned_model}")
                print(f"Fine-tuning job completed successfully. Fine-tuned model: {fine_tuned_model}")
                break
            elif status in ['failed', 'cancelled']:
                logger.error(f"Fine-tuning job ended with status: {status}")
                print(f"Error: Fine-tuning job ended with status: {status}")
                break

            # Wait before checking again
            time.sleep(30)  # Wait for 30 seconds
    except openai.APIError as e:
        logger.error(f"OpenAI API error while monitoring fine-tuning job: {e}")
        print(f"Error: OpenAI API error while monitoring fine-tuning job: {e}")
    except KeyboardInterrupt:
        logger.warning("Fine-tuning job monitoring interrupted by user.")
        print("Fine-tuning job monitoring interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error while monitoring fine-tuning job: {e}")
        print(f"Error: Unexpected error while monitoring fine-tuning job: {e}")

    return fine_tuned_model

def evaluate_model(client: OpenAI, fine_tuned_model: str, test_file_path: str):
    """
    Evaluates the fine-tuned model on the test dataset.

    Args:
        client (OpenAI): The OpenAI client instance.
        fine_tuned_model (str): The name of the fine-tuned model.
        test_file_path (str): Path to the test dataset JSONL file.
    """
    logger.info(f"Evaluating fine-tuned model '{fine_tuned_model}' on test dataset.")
    print(f"Evaluating fine-tuned model '{fine_tuned_model}' on test dataset.")

    # Check if test file exists
    if not os.path.isfile(test_file_path):
        logger.error(f"Test file not found at path: {test_file_path}")
        print(f"Error: Test file not found at path: {test_file_path}")
        return

    # Read test dataset
    test_data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # Evaluate each test example
    total = len(test_data)
    logger.info(f"Total test examples: {total}")
    print(f"Total test examples: {total}")

    for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
        messages = example['messages'][:-1]  # All messages except the assistant's response
        expected_response = example['messages'][-1]['content']  # The assistant's response

        # Generate completion using the fine-tuned model
        try:
            response = client.chat.completions.create(
                model=fine_tuned_model,
                messages=messages,
                temperature=0.0,  # Deterministic output
            )
            generated_response = response.choices[0].message.content.strip()

            # Log the expected and generated responses
            logger.info(f"Test example {idx + 1}/{total}")
            logger.info(f"User prompt: {messages[-1]['content']}")
            logger.info(f"Expected response: {expected_response}")
            logger.info(f"Generated response: {generated_response}")


        except openai.APIError as e:
            logger.error(f"OpenAI API error during evaluation at index {idx}: {e}")
            print(f"Error: OpenAI API error during evaluation at index {idx}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during evaluation at index {idx}: {e}")
            print(f"Error: Unexpected error during evaluation at index {idx}: {e}")

    logger.info("Evaluation completed.")
    print("Evaluation completed.")

if __name__ == '__main__':
    main()
