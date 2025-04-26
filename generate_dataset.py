# generate_dataset.py

import os
import json
import io
import random
import shutil
from tqdm import tqdm
from collections import defaultdict
from cmsis_svd.parser import SVDParser
from multiprocessing import Pool, cpu_count
import tiktoken  # For token counting
import numpy as np

def main():
    # Configuration settings
    config = {
        'svd_files': 'cmsis-svd-data/data/',  # Directory containing SVD files
        'output_dir': './datasets',           # Output directory for datasets
        'system_prompt': 'You are an expert on microcontrollers and can provide detailed information about their peripherals, registers, and fields.',
        'max_budget': 240,                    # Maximum budget in dollars
        'safety_margin': 10,                  # Safety margin in dollars to stay under budget
        'model_name': 'gpt-4o-2024-08-06',    # Model name for pricing
        'target_epochs': 4,                   # Number of epochs for fine-tuning
    }

    print("Starting dataset generation script...")

    # Ensure the output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)

    # Collect SVD files
    print("\nStep 1: Collecting SVD files...")
    svd_files = collect_svd_files(config['svd_files'])
    print(f"Total SVD files found: {len(svd_files)}")

    # Shuffle SVD files to randomize the order
    random.shuffle(svd_files)

    # Initialize variables
    total_tokens = 0
    total_cost = 0
    dataset_size_bytes = 0
    max_allowed_cost = config['max_budget'] - config['safety_margin']
    encoding = tiktoken.get_encoding("cl100k_base")

    # Open the main dataset file for writing
    main_data_path = os.path.join(config['output_dir'], 'main_data.jsonl')
    f_dataset = open(main_data_path, 'w', encoding='utf-8')

    print("\nStep 2: Processing SVD files and generating dataset...")
    print(f"Building dataset until estimated cost reaches ${max_allowed_cost:.2f}")

    # Process SVD files one by one
    for svd_file_path in tqdm(svd_files, desc="Processing SVD files"):
        try:
            # Process the SVD file to generate conversations
            svd_content = read_svd_file(svd_file_path)
            svd_json_content = svd_to_json(svd_content)
            svd_json = json.loads(svd_json_content)
            conversations = generate_conversation_examples(svd_json['device'], config['system_prompt'])

            # For each conversation
            for conversation in conversations:
                # Estimate number of tokens
                messages = conversation["messages"]
                num_tokens = num_tokens_from_messages(messages, encoding)
                # Update total tokens
                total_tokens += num_tokens
                # Estimate cost
                estimated_cost = estimate_fine_tuning_cost(
                    total_tokens,
                    config['model_name'],
                    config['target_epochs']
                )
                # Check if estimated cost exceeds budget
                if estimated_cost >= max_allowed_cost:
                    total_tokens -= num_tokens  # Revert last addition
                    break  # Stop adding more conversations

                # Write conversation to dataset file
                json_line = json.dumps(conversation)
                f_dataset.write(json_line + '\n')
                dataset_size_bytes += len(json_line.encode('utf-8')) + 1  # +1 for newline character

                # Print current stats
                print(f"Total Tokens: {total_tokens}, Estimated Cost: ${estimated_cost:.2f}, Dataset Size: {dataset_size_bytes / (1024 * 1024):.2f} MB", end='\r')

            # If estimated cost reached, break out of outer loop
            if estimated_cost >= max_allowed_cost:
                print("\nEstimated cost has reached the budget limit.")
                break

        except Exception as e:
            print(f"Error processing {svd_file_path}: {e}")

    # Close the dataset file
    f_dataset.close()

    # Print final stats
    print(f"\n\nDataset generation completed.")
    print(f"Total Tokens: {total_tokens}")
    print(f"Estimated Fine-Tuning Cost: ${estimated_cost:.2f}")
    print(f"Dataset Size: {dataset_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Dataset saved to {main_data_path}")

    # Split the dataset into training, testing, and validation sets
    print("\nStep 3: Splitting dataset into training, testing, and validation sets...")
    split_datasets(main_data_path, config['output_dir'])
    print("\nDataset generation complete!")

def collect_svd_files(svd_files_directory):
    """
    Collects all SVD file paths from the directory.

    :param svd_files_directory: Directory containing SVD files organized by vendor
    :return: List of SVD file paths
    """
    svd_files = []
    for root, dirs, files in os.walk(svd_files_directory):
        for file in files:
            if file.endswith('.svd'):
                file_path = os.path.join(root, file)
                svd_files.append(file_path)
    return svd_files

def read_svd_file(file_path):
    """
    Reads the content of an SVD file.

    :param file_path: Path to the SVD file
    :return: Content of the SVD file
    """
    with open(file_path, 'rb') as f:
        content = f.read()
    return content

def svd_to_json(svd_content):
    """
    Converts SVD content to a JSON string using the CMSIS-SVD parser.

    :param svd_content: SVD file content in bytes
    :return: JSON string representing the SVD content
    """
    if isinstance(svd_content, bytes):
        svd_content = svd_content.decode("utf-8")
    else:
        # Return as is if already a JSON string
        return svd_content

    svd_file = io.StringIO(svd_content)
    parser = SVDParser.for_xml_file(svd_file)
    device = parser.get_device()

    device_dict = {
        "device": {
            "name": device.name,
            "description": device.description,
            "peripherals": []
        }
    }

    for peripheral in device.peripherals:
        peripheral_info = {
            "name": peripheral.name,
            "description": peripheral.description,
            "base_address": hex(peripheral.base_address),
            "registers": []
        }

        for register in peripheral.registers:
            register_info = {
                "name": register.name,
                "address_offset": hex(register.address_offset),
                "size": register.size,
                "description": register.description,
                "fields": []
            }

            for field in register.fields:
                field_info = {
                    "name": field.name,
                    "bit_offset": field.bit_offset,
                    "bit_width": field.bit_width,
                    "description": field.description
                }
                register_info["fields"].append(field_info)

            peripheral_info["registers"].append(register_info)

        device_dict["device"]["peripherals"].append(peripheral_info)

    return json.dumps(device_dict, indent=4)

def generate_conversation_examples(device, system_prompt):
    """
    Generates conversation examples based on the device information.

    :param device: Dictionary containing device information
    :param system_prompt: The system prompt to include in each conversation
    :return: List of conversation examples in the required format
    """
    conversations = []
    name = device.get('name', 'Unknown Device')

    # Basic device-level questions
    if device.get('description', ''):
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"What is the description of {name} microcontroller?"},
                {"role": "assistant", "content": device.get('description', '')}
            ]
        }
        conversations.append(conversation)

    peripherals = device.get('peripherals', [])
    if peripherals:
        peripheral_names = ', '.join([peripheral['name'] for peripheral in peripherals])
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"List all the peripherals for {name} microcontroller."},
                {"role": "assistant", "content": peripheral_names}
            ]
        }
        conversations.append(conversation)

        for peripheral in peripherals:
            # Peripheral-level questions
            p_name = peripheral['name']
            if peripheral.get('description', ''):
                conversation = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"What is the description of {p_name} peripheral for {name} microcontroller?"},
                        {"role": "assistant", "content": peripheral.get('description', '')}
                    ]
                }
                conversations.append(conversation)

            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"What is the base address of {p_name} peripheral for {name} microcontroller?"},
                    {"role": "assistant", "content": peripheral.get('base_address', '')}
                ]
            }
            conversations.append(conversation)

            registers = peripheral.get('registers', [])
            if registers:
                register_names = ', '.join([register['name'] for register in registers])
                conversation = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"List all the registers of {p_name} peripheral for {name} microcontroller."},
                        {"role": "assistant", "content": register_names}
                    ]
                }
                conversations.append(conversation)

                for register in registers:
                    # Register-level questions
                    r_name = register["name"]
                    if register.get('description', ''):
                        conversation = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"What is the description of {r_name} register from {p_name} peripheral for {name} microcontroller?"},
                                {"role": "assistant", "content": register.get('description', '')}
                            ]
                        }
                        conversations.append(conversation)

                    conversation = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"What is the size of {r_name} register from {p_name} peripheral for {name} microcontroller?"},
                            {"role": "assistant", "content": str(register.get('size', ''))}
                        ]
                    }
                    conversations.append(conversation)

                    conversation = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"What is the address_offset of {r_name} register from {p_name} peripheral for {name} microcontroller?"},
                            {"role": "assistant", "content": register.get('address_offset', '')}
                        ]
                    }
                    conversations.append(conversation)

                    fields = register.get('fields', [])
                    if fields:
                        field_names = ', '.join([field['name'] for field in fields])
                        conversation = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"List all the fields of {r_name} register from {p_name} peripheral for {name} microcontroller."},
                                {"role": "assistant", "content": field_names}
                            ]
                        }
                        conversations.append(conversation)

                        for field in fields:
                            # Field-level questions
                            f_name = field["name"]
                            if field.get('description', ''):
                                conversation = {
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"What is the description of {f_name} field from {r_name} register from {p_name} microcontroller?"},
                                        {"role": "assistant", "content": field.get('description', '')}
                                    ]
                                }
                                conversations.append(conversation)

                            if "bit_offset" in field:
                                conversation = {
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"What is the bit_offset of {f_name} field from {r_name} register from {p_name} microcontroller?"},
                                        {"role": "assistant", "content": str(field.get('bit_offset', ''))}
                                    ]
                                }
                                conversations.append(conversation)

                            if "bit_width" in field:
                                conversation = {
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"What is the bit_width of {f_name} field from {r_name} register from {p_name} microcontroller?"},
                                        {"role": "assistant", "content": str(field.get('bit_width', ''))}
                                    ]
                                }
                                conversations.append(conversation)
    return conversations

def num_tokens_from_messages(messages, encoding, tokens_per_message=3, tokens_per_name=1):
    """
    Estimates the number of tokens used by a list of messages.

    :param messages: List of messages
    :param encoding: tiktoken encoding
    :return: Number of tokens
    """
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key == "name":
                num_tokens += tokens_per_name
            value = value or ""
            num_tokens += len(encoding.encode(value))
    num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def estimate_fine_tuning_cost(total_tokens, model_name, target_epochs):
    """
    Estimates the fine-tuning cost based on the total tokens and pricing.

    :param total_tokens: Total number of tokens
    :param model_name: Name of the model for pricing
    :param target_epochs: Number of epochs
    :return: Estimated cost in dollars
    """
    # Pricing for models (replace with actual model name and pricing if different)
    pricing = {
        'gpt-4o-2024-08-06': {
            'training': 25.00 / 1_000_000,  # $25.00 per 1M training tokens
        },
    }

    if model_name not in pricing:
        raise ValueError(f"Pricing for model '{model_name}' not found.")

    model_pricing = pricing[model_name]
    total_training_tokens = total_tokens * target_epochs
    cost = total_training_tokens * model_pricing['training']
    return cost

def split_datasets(main_data_path, output_dir):
    """
    Splits the main dataset into training, testing, and validation datasets.
    Assumes the main dataset is in JSON Lines format.

    :param main_data_path: Path to the main dataset file
    :param output_dir: Directory to save the split datasets
    """
    train_data_path = os.path.join(output_dir, 'train_data.jsonl')
    test_data_path = os.path.join(output_dir, 'test_data.jsonl')
    val_data_path = os.path.join(output_dir, 'validation_data.jsonl')

    train_file = open(train_data_path, 'w', encoding='utf-8')
    test_file = open(test_data_path, 'w', encoding='utf-8')
    val_file = open(val_data_path, 'w', encoding='utf-8')

    train_count = test_count = val_count = 0

    print("Counting total lines in main dataset...")
    total_lines = sum(1 for _ in open(main_data_path, 'r', encoding='utf-8'))
    print(f"Total conversations in main dataset: {total_lines}")

    with open(main_data_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc="Splitting dataset", total=total_lines):
            r = random.random()
            if r < 0.8:
                train_file.write(line)
                train_count += 1
            elif r < 0.9:
                test_file.write(line)
                test_count += 1
            else:
                val_file.write(line)
                val_count += 1

    train_file.close()
    test_file.close()
    val_file.close()

    print(f"\nTraining dataset saved to {train_data_path} ({train_count} samples)")
    print(f"Testing dataset saved to {test_data_path} ({test_count} samples)")
    print(f"Validation dataset saved to {val_data_path} ({val_count} samples)")

if __name__ == "__main__":
    main()
