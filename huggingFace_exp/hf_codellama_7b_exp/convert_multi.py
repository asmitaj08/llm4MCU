import json
import os
from pathlib import Path
from multiprocessing import get_context
from tqdm import tqdm

def convert_chat_messages_to_codellama_format(input_path):
    """
    Converts a ChatGPT-style dataset to Code LLaMA `[INST]` format and saves it
    with `_for_codellama` appended to the original filename.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return

    # Prepare output filename
    output_name = input_path.stem + "_for_codellama" + input_path.suffix
    output_path = input_path.parent / output_name

    with input_path.open('r', encoding='utf-8') as infile, output_path.open('w', encoding='utf-8') as outfile:
        for line in infile:
            obj = json.loads(line)
            messages = obj.get("messages", [])

            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            prompt = (
                f"[INST] <<SYS>> {system_msg.strip()} <</SYS>> "
                f"{user_msg.strip()} [/INST] {assistant_msg.strip()}"
            )

            json.dump({"text": prompt}, outfile)
            outfile.write("\n")
    
    os.remove(input_path)
    print(f" Conversion complete! Saved to: {output_path}")


def process_file(input_path_str):
    input_path = Path(input_path_str)
    if input_path.exists():
        convert_chat_messages_to_codellama_format(input_path)
    else:
        print(f"File not found: {input_path}")

if __name__ == "__main__":
    print("ChatGPT → Code LLaMA Format Converter")
    # input_file = input("🔹 Enter the path to your input JSONL file: ").strip()
    main_dir = Path("./evaluation_mcu_svd_dataset_codellama/")

    input_files = []

    # Collect all input files
    for sub_dir in main_dir.iterdir():
        if sub_dir.is_dir():
            for filename in ["main_data.jsonl","test_data.jsonl", "train_data.jsonl", "validation_data.jsonl"]:
                file_path = sub_dir / filename
                if file_path.exists():
                    input_files.append(str(file_path))

    print(f"Total files to process: {len(input_files)}")

    # Parallel Processing: 14 workers (out of 16 CPUs allocated on HPC)
    cpu_workers = 14

    with get_context("spawn").Pool(processes=cpu_workers) as pool:
        list(tqdm(pool.imap_unordered(process_file, input_files), total=len(input_files), desc="Processing files"))