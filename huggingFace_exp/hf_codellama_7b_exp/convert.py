import json
import os
from pathlib import Path

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

    print(f" Conversion complete! Saved to: {output_path}")

if __name__ == "__main__":
    print("ChatGPT → Code LLaMA Format Converter")
    input_file = input("🔹 Enter the path to your input JSONL file: ").strip()
    convert_chat_messages_to_codellama_format(input_file)