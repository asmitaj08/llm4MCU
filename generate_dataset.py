import os
import json
import io
import random
import shutil
from tqdm import tqdm
from collections import defaultdict
from cmsis_svd.parser import SVDParser
from multiprocessing import Pool, cpu_count

def main():
    # Configuration settings
    config = {
        'svd_files': 'cmsis-svd-data/data/',  # Directory containing SVD files
        'output_dir': './datasets',           # Output directory for datasets
        'system_prompt': 'You are an expert on microcontrollers and can provide detailed information about their peripherals, registers, and fields.',
    }

    print("Starting dataset generation script...")

    # Ensure the output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)

    # Build mappings from vendors to SVD files and from SVD files to their content
    print("\nStep 1: Building SVD file mappings...")
    vendor_to_svd, svd_to_vendor, svd_to_content = build_svd_mappings(config['svd_files'])
    print(f"Found {len(svd_to_content)} SVD files.")

    # Process all SVD files and generate conversation examples in parallel
    print("\nStep 2: Processing SVD files and generating conversation examples...")
    temp_dir = os.path.join(config['output_dir'], 'temp_qa_files')
    process_svd_files(svd_to_content, temp_dir, config['system_prompt'])

    # Merge temporary QA files into main dataset
    print("\nStep 3: Merging temporary files into main dataset...")
    main_data_path = os.path.join(config['output_dir'], 'main_data.jsonl')
    merge_temp_files(temp_dir, main_data_path)
    print(f"Main dataset saved to {main_data_path}")

    # Clean up temporary directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)

    # Split the dataset into training, testing, and validation sets
    print("\nStep 4: Splitting dataset into training, testing, and validation sets...")
    split_datasets(main_data_path, config['output_dir'])
    print("\nDataset generation complete!")

def build_svd_mappings(svd_files_directory):
    """
    Builds mappings for vendors to SVD files, SVD files to vendors, and SVD files to their content.

    :param svd_files_directory: Directory containing SVD files organized by vendor
    :return: Tuple containing vendor_to_svd, svd_to_vendor, svd_to_content dictionaries
    """
    vendor_to_svd = defaultdict(list)
    svd_to_vendor = {}
    svd_to_content = {}

    print("Collecting SVD files...")
    # First, collect all SVD file paths
    svd_files = []
    for root, dirs, files in os.walk(svd_files_directory):
        for file in files:
            if file.endswith('.svd'):
                file_path = os.path.join(root, file)
                svd_files.append((file_path, root))

    print(f"Total SVD files found: {len(svd_files)}")

    # Now process the SVD files with a progress bar
    for file_info in tqdm(svd_files, desc="Reading SVD files"):
        file_path, root = file_info
        subdirs = root.split(os.sep)
        if len(subdirs) > 2:
            vendor = subdirs[2]  # Get the vendor name
        else:
            vendor = "Unknown Vendor"
        file = os.path.basename(file_path)
        vendor_to_svd[vendor].append(file)
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            svd_to_content[file] = content
            svd_to_vendor[file] = vendor
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return vendor_to_svd, svd_to_vendor, svd_to_content

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
                                        {"role": "user", "content": f"What is the description of {f_name} field from {r_name} register from {p_name} peripheral for {name} microcontroller?"},
                                        {"role": "assistant", "content": field.get('description', '')}
                                    ]
                                }
                                conversations.append(conversation)

                            if "bit_offset" in field:
                                conversation = {
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"What is the bit_offset of {f_name} field from {r_name} register from {p_name} peripheral for {name} microcontroller?"},
                                        {"role": "assistant", "content": str(field.get('bit_offset', ''))}
                                    ]
                                }
                                conversations.append(conversation)

                            if "bit_width" in field:
                                conversation = {
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"What is the bit_width of {f_name} field from {r_name} register from {p_name} peripheral for {name} microcontroller?"},
                                        {"role": "assistant", "content": str(field.get('bit_width', ''))}
                                    ]
                                }
                                conversations.append(conversation)
    return conversations

def process_single_svd(args):
    """
    Worker function to process a single SVD file and write conversation examples to a temporary file.

    :param args: Tuple containing svd_name, svd_content, temp_dir, and system_prompt
    :return: True if successful, False otherwise
    """
    svd_name, svd, temp_dir, system_prompt = args
    try:
        svd_content = svd_to_json(svd)
        svd_json = json.loads(svd_content)
        conversations = generate_conversation_examples(svd_json['device'], system_prompt)
        # Write conversations to a temporary file in JSON Lines format
        safe_svd_name = svd_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        temp_file_path = os.path.join(temp_dir, f'qa_{safe_svd_name}.jsonl')
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                json_line = json.dumps(conv)
                f.write(json_line + '\n')
        return True
    except Exception as e:
        print(f"Error processing {svd_name}: {e}")
        return False

def process_svd_files(svd_to_content, temp_dir, system_prompt):
    """
    Processes all SVD files and generates conversation examples in parallel.
    Each SVD file's conversations are written to a separate temporary file.

    :param svd_to_content: Dictionary mapping SVD filenames to their content
    :param temp_dir: Temporary directory to store individual QA files
    :param system_prompt: The system prompt to include in each conversation
    """
    # Prepare arguments for processing
    svd_items = [(svd_name, svd, temp_dir, system_prompt) for svd_name, svd in svd_to_content.items()]

    # Ensure the temporary directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Use multiprocessing Pool to process SVD files in parallel
    total_files = len(svd_items)
    print(f"Total SVD files to process: {total_files}")
    print(f"Processing SVD files using {cpu_count()} cores...")

    with Pool() as pool:
        # tqdm needs to be at the outer scope to update properly
        with tqdm(total=total_files, desc="Processing SVD files") as pbar:
            for _ in pool.imap_unordered(process_single_svd, svd_items):
                pbar.update()

def merge_temp_files(temp_dir, output_file):
    """
    Merges all temporary QA files into a single main dataset file in JSON Lines format.

    :param temp_dir: Directory containing temporary QA files
    :param output_file: Path to the output main dataset file
    """
    import glob
    temp_files = glob.glob(os.path.join(temp_dir, 'qa_*.jsonl'))
    print(f"Total temporary files to merge: {len(temp_files)}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for temp_file in tqdm(temp_files, desc="Merging files"):
            with open(temp_file, 'r', encoding='utf-8') as infile:
                shutil.copyfileobj(infile, outfile)

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
