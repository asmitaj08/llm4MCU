# ────────────────────────────────────────────────────────────────────────
# Import necessary libraries
# ────────────────────────────────────────────────────────────────────────
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# ────────────────────────────────────────────────────────────────────────
# Load tokenizer and model
# ────────────────────────────────────────────────────────────────────────

# Path to your fine-tuned model
model_dir = "codellama_7b/final_model"
base_model_id = "codellama/CodeLlama-7b-Instruct-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load base model with quantization config
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

# Load LoRA adapter
model = PeftModel.from_pretrained(model, model_dir)
model.eval()

# ────────────────────────────────────────────────────────────────────────
# Load the pre-trained Sentence Transformer model
# ────────────────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ────────────────────────────────────────────────────────────────────────
# Function to compute similarity
# ────────────────────────────────────────────────────────────────────────
def compute_similarity(model_output: str, ground_truth: str) -> tuple:
    embeddings = embedding_model.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    percentage = cos_sim * 100
    return cos_sim, percentage

# ────────────────────────────────────────────────────────────────────────
# Function to query the model
# ────────────────────────────────────────────────────────────────────────
def ask_bot(question: str) -> str:
    formatted_prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Optionally post-process
    return generated_text.replace(formatted_prompt, "").strip()


# ────────────────────────────────────────────────────────────────────────
# Evaluation function
# ────────────────────────────────────────────────────────────────────────
def evaluate(jsonl_file_path):
    scores = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            full_text = data["text"]

            # Split into prompt and ground truth
            if '[/INST]' in full_text:
                prompt_part, sep, answer_part = full_text.partition('[/INST]')
                question = prompt_part + sep  # keep [INST]...[/INST]
                ground_truth = answer_part.strip()
            else:
                # fallback if no [/INST] (just in case)
                question = full_text
                ground_truth = ""

            model_output = ask_bot(question)

            cos_sim, percentage = compute_similarity(model_output, ground_truth)

            scores.append({
                'question': question,
                'ground_truth': ground_truth,
                'model_output': model_output,
                'cosine_similarity': cos_sim,
                'similarity_percentage': percentage
            })

    return scores


# ────────────────────────────────────────────────────────────────────────
# Run evaluation
# ────────────────────────────────────────────────────────────────────────
results = evaluate("../datasets/test_data_for_codellama.jsonl")


for result in results:
    print(f"Question: {result['question']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Model Output: {result['model_output']}")
    print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
    print(f"Similarity Percentage: {result['similarity_percentage']:.2f}%\n")

average_similarity = sum([r['cosine_similarity'] for r in results]) / len(results)
average_percentage = average_similarity * 100
print(f"Average Cosine Similarity: {average_similarity:.4f}")
print(f"Average Similarity Percentage: {average_percentage:.2f}%")
