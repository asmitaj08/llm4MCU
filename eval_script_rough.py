# Import necessary libraries
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your finetuned LLM model here
# For example:
# model = YourLLMModel.load_from_checkpoint('path_to_checkpoint')

# Load the pre-trained Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(model_output: str, ground_truth: str) -> tuple:
    # Same as before
    embeddings = embedding_model.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    percentage = cos_sim * 100
    return cos_sim, percentage

# def ask_bot(question: str) -> str:
#     # Use your actual model's inference method
#     response = model.generate(question)
#     return response

def evaluate(json_q_a_file_path="./nrf52840.json"):
    # Same as before
    with open(json_q_a_file_path, 'r') as file:
        q_a = json.load(file)

    scores = []
    for q_a_pair in q_a:
        question = q_a_pair["prompt"]
        ground_truth = q_a_pair["completion"]

        # Get the model's response
        model_output = ask_bot(question)

        # Compute similarity
        cos_sim, percentage = compute_similarity(model_output, ground_truth)

        # Append the results
        scores.append({
            'question': question,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'cosine_similarity': cos_sim,
            'similarity_percentage': percentage
        })

    return scores

# Run the evaluation
results = evaluate("./nrf52840.json")

# Display results
for result in results:
    print(f"Question: {result['question']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Model Output: {result['model_output']}")
    print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
    print(f"Similarity Percentage: {result['similarity_percentage']:.2f}%\n")

# Compute average similarity
average_similarity = sum([r['cosine_similarity'] for r in results]) / len(results)
average_percentage = average_similarity * 100
print(f"Average Cosine Similarity: {average_similarity:.4f}")
print(f"Average Similarity Percentage: {average_percentage:.2f}%")
