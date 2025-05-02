# inference_codellama.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# ───────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────
model_dir = "codellama_7b/final_model"  # relative path from finetuned_models/
base_model_id = "codellama/CodeLlama-7b-Instruct-hf"

# ───────────────────────────────────────────────────────────────────────
# LOAD TOKENIZER & BASE MODEL
# ───────────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# ───────────────────────────────────────────────────────────────────────
# LOAD LoRA ADAPTER
# ───────────────────────────────────────────────────────────────────────
print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(model, model_dir)
model.eval()

# ───────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTION
# ───────────────────────────────────────────────────────────────────────
def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ─── Trim output after first [/INST]
    if '[/INST]' in generated_text:
        parts = generated_text.split('[/INST]', 1)
        answer_only = parts[1].strip()
    else:
        answer_only = generated_text.strip()

    # ─── Further trim at first period
    if '.' in answer_only:
        first_sentence = answer_only.split('.', 1)[0].strip()
    else:
        first_sentence = answer_only.strip()

    return first_sentence

# ───────────────────────────────────────────────────────────────────────
# RUN INTERACTIVE PROMPT
# ───────────────────────────────────────────────────────────────────────
print("\nModel loaded! Type a prompt and press Enter (or type 'exit' to quit)\n")

while True:
    user_input = input(">> ")
    if user_input.strip().lower() == "exit":
        break

    formatted_prompt = f"[INST] <<SYS>> You are an expert on microcontrollers and can provide detailed information about their peripherals, registers, and fields. <</SYS>> {user_input} [/INST]"

    output = generate_response(formatted_prompt)
    print(f"\nGenerated Answer:\n{output}\n")
