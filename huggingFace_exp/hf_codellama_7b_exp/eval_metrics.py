import pandas as pd
import re
import multiprocessing as mp
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
from bert_score import score as bertscore
import os
import sys
import torch

# pip install pandas sentence-transformers scikit-learn python-Levenshtein bert-score


# ==== Metrics ====

# Cosine similarity
eval_embedding = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
def compute_cosine(prediction, ground_truth):
    embeddings = eval_embedding.encode([str(prediction), str(ground_truth)])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Exact match after normalization
# Check if the ground truth exists as a substring in the LLM's output after normalization (removing punctuation, lowercasing, etc.).
# Useful for matching exact values, addresses, register names even if LLM adds extra text.
def safe_normalize(text):
    if pd.isna(text):
        text = ""
    text = str(text).lower().strip()
    return re.sub(r'[^a-z0-9]', '', text)

def exact_match_normalized(pred, gt):
    return safe_normalize(gt) in safe_normalize(pred)

# Token-level F1
# Precision & Recall overlap between predicted tokens and ground truth tokens.
# Useful when order doesn't matter (e.g., lists of registers).
def token_f1(pred, gt):
    pred_tokens = str(pred).lower().split()
    gt_tokens = str(gt).lower().split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# Levenshtein similarity (character edit distance)
# Measures how many edits (insert, delete, substitute) are needed to convert prediction to ground truth.
# Useful for addresses, hex values where small differences matter.
def normalized_levenshtein(pred, gt):
    pred = str(pred)
    gt = str(gt)
    distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    return 1 - distance / max_len


# Numeric match
# Extract numeric values (e.g., hex, decimal) from both prediction & ground truth, and compare.
# Useful for answers that are supposed to be addresses or specific values.
def numeric_match(pred, gt):
    pred = str(pred)
    gt = str(gt)
    numbers_pred = set(re.findall(r'\b0x[0-9a-fA-F]+\b|\b\d+\b', pred))
    numbers_gt = set(re.findall(r'\b0x[0-9a-fA-F]+\b|\b\d+\b', gt))
    return numbers_gt.issubset(numbers_pred)


# Batched BERTScore on GPU ((Semantic Similarity))
# Token-level contextual similarity (better than cosine on sentence level for factual QA).
def compute_bertscore_batch(predictions, ground_truths, batch_size=64, device='cuda'):
    predictions = [str(p) for p in predictions]
    ground_truths = [str(g) for g in ground_truths]
    all_scores = []
    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i+batch_size]
        batch_gts = ground_truths[i:i+batch_size]
        _, _, F1 = bertscore(batch_preds, batch_gts, lang='en', device=device)
        all_scores.extend(F1.tolist())
    return all_scores


# Weighted scoring
# Weighted_Score = (
#     0.3 * E +            # Exact match is top priority
#     0.25 * N +           # Numeric match is nearly as important
#     0.2 * F1 +           # Token overlap matters for lists
#     0.15 * BERT +        # BERTScore for semantic correctness
#     0.05 * COS +         # Cosine similarity (already lower than BERT)
#     0.05 * LEV           # Levenshtein for formatting quirks
# )
#  This gives:
# 55% weight to factual correctness (E + N)
# 20% to token-level overlap (F1)
# 20% to semantic similarity (BERT + COS)
# 5% to string edit accuracy (LEV)

def compute_weighted_score(metrics):
    return (
        0.3 * float(metrics['exact_match']) +
        0.25 * float(metrics['numeric_match']) +
        0.2 * metrics['token_f1'] +
        0.15 * metrics['bertscore_f1'] +
        0.05 * metrics['cosine_sim'] +
        0.05 * metrics['levenshtein_sim']
    )

# Classification (optional)
# For usability, we might want:
# Pass if weighted_score >= 0.7 (high accuracy)
# Partial if 0.5 <= score < 0.7
# Fail if < 0.5
def classify_score(score):
    if score >= 0.7:
        return "Pass"
    elif score >= 0.5:
        return "Partial"
    return "Fail"

# CPU metrics per pair
def compute_cpu_metrics(args):
    prediction, ground_truth = args
    return {
        "cosine_sim": compute_cosine(prediction, ground_truth),
        "exact_match": exact_match_normalized(prediction, ground_truth),
        "numeric_match": numeric_match(prediction, ground_truth),
        "token_f1": token_f1(prediction, ground_truth),
        "levenshtein_sim": normalized_levenshtein(prediction, ground_truth),
    }

# Evaluation per answer column
def evaluate_column(df, ans_col, gt_col, batch_size=64, workers=16):
    args = list(zip(df[ans_col], df[gt_col]))
    print(f"Computing CPU metrics for {ans_col}...")
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        cpu_results = list(tqdm(pool.imap(compute_cpu_metrics, args), total=len(args)))
    cpu_metrics_df = pd.DataFrame(cpu_results)

    print(f"Computing BERTScore for {ans_col} (batch size {batch_size})...")
    bert_scores = compute_bertscore_batch(df[ans_col].tolist(), df[gt_col].tolist(), batch_size=batch_size, device='cuda')
    cpu_metrics_df['bertscore_f1'] = bert_scores

    cpu_metrics_df['weighted_score'] = cpu_metrics_df.apply(compute_weighted_score, axis=1)
    cpu_metrics_df['classification'] = cpu_metrics_df['weighted_score'].apply(classify_score)

    # Rename columns to match ans_col prefix
    cpu_metrics_df = cpu_metrics_df.add_prefix(f'{ans_col}_')

    return cpu_metrics_df

# Full multi-answer evaluation
def evaluate_multi_answers(df, answer_cols, gt_col, batch_size=64, workers=16):
    result_dfs = []
    for ans_col in answer_cols:
        result_df = evaluate_column(df, ans_col, gt_col, batch_size=batch_size, workers=workers)
        result_dfs.append(result_df)
    combined_metrics_df = pd.concat(result_dfs, axis=1)
    return pd.concat([df, combined_metrics_df], axis=1)

# ==== Main Execution ====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--gt_column", type=str, default="ground_truth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    answer_cols = ['baseline model_output', 'few shot model_output', 'ft reg model_output','ft few shot model_output'] #openAI results 
    # answer_cols = ['baseline model_output', 'few shot model_output', 'ft model output', 'ft few_shot model output']


    # 1. Check if file exists
    if not os.path.isfile(args.input_csv):
        print(f"ERROR: Input CSV file '{args.input_csv}' does not exist.")
        sys.exit(1)

    # 2. Load CSV header only (fast)
    try:
        df_sample = pd.read_csv(args.input_csv, nrows=0)
    except Exception as e:
        print(f"ERROR: Failed to read CSV file '{args.input_csv}': {e}")
        sys.exit(1)

    # 3. Check for required columns
    missing_cols = [col for col in answer_cols if col not in df_sample.columns]
    if missing_cols:
        print(f"ERROR: Missing columns in CSV: {missing_cols}")
        sys.exit(1)

    print(f"Input CSV '{args.input_csv}' validated successfully with required columns: {answer_cols}")

    df = pd.read_csv(args.input_csv)
    evaluated_df = evaluate_multi_answers(df, answer_cols, gt_col=args.gt_column, batch_size=args.batch_size, workers=args.workers)
    evaluated_df.to_csv(args.output_csv, index=False)
    print(f"Saved evaluated CSV to {args.output_csv}")
