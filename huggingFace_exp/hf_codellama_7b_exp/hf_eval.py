import pickle
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
import base64
import os
from PIL import Image
import pytesseract
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
import uuid
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import io
import re
from IPython.display import HTML, display
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import random
from peft import PeftModel


load_dotenv()
base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
# embedd_model = "sentence-transformers/all-MiniLM-L6-v2" #will have to experiment with embeddig models
# embedd_model = "intfloat/e5-large-v2"
embedd_model = "BAAI/bge-large-en-v1.5"

embedding = HuggingFaceEmbeddings(model_name=embedd_model)

eval_embedding = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

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

# # Create Hugging Face pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     pad_token_id=model.config.eos_token_id, #avoiding warning Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
#     max_new_tokens=512,
#     # temperature=0.0, # no need as do_sample=False
#     do_sample=False,
#     return_full_text=False
# )
# # Wrap it in LangChain-compatible interface
# model_pipe = HuggingFacePipeline(pipeline=pipe)

##ft 
ft_model_dir = "codellama_7b/final_model"
assert os.path.exists(ft_model_dir), f"Finetuned Model directory not found: {ft_model_dir}"
# Load LoRA adapter
ft_model = PeftModel.from_pretrained(model, ft_model_dir)
# # Create Hugging Face pipeline for finetuned model
# pipe_ft = pipeline(
#     "text-generation",
#     model=ft_model,
#     tokenizer=tokenizer,
#     pad_token_id=model.config.eos_token_id, #avoiding warning Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
#     max_new_tokens=512,
#     # temperature=0.0, # no need as do_sample=False
#     do_sample=False,
#     return_full_text=False
# )
# # Wrap it in LangChain-compatible interface
# model_pipe_ft = HuggingFacePipeline(pipeline=pipe_ft)

# def ask_bot(chain_multimodal_rag, query):
#     raw_output = chain_multimodal_rag.invoke(query)
#     # print(f"Raw output : {raw_output}\n")
#     # print("*********************")
#     for line in raw_output.strip().splitlines():
#         if line.strip():
#             return line.strip()
#     return raw_output.strip()

# def ask_bot_ft(question: str) -> str:
#     formatted_prompt = f"### Instruction:\n{question}\n\n### Response:\n"
#     inputs = tokenizer(formatted_prompt, return_tensors="pt").to(ft_model.device)
#     with torch.no_grad():
#         outputs = ft_model.generate(
#             **inputs,
#             max_new_tokens=512,
#             temperature=0.7,
#             top_p=0.95,
#             do_sample=True
#         )
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     answer = generated_text.replace(formatted_prompt, "").strip()
#     # Optionally post-process
#     # return generated_text.replace(formatted_prompt, "").strip()
#     # Normalize: clean each line
#     lines = [
#         re.sub(r"[.,:;!?]+$", "", line.strip())  # Remove trailing punctuation
#         for line in answer.splitlines()
#         if line.strip()
#     ]

#     return ", ".join(lines)


def ask_bot(retriever_multi_vector_img, question, curr_model,few_shot_examples=None) -> str:
    # Step 1: Retrieve relevant summaries (text + table + image) via RAG
    docs = retriever_multi_vector_img.get_relevant_documents(question)
    summaries = [
    doc if isinstance(doc, str) else doc.page_content for doc in docs
   ]

    # system_prompt = (
    #     "You are an expert on microcontrollers and can provide detailed information about their peripherals, registers, and fields.\n"
    #     "Answer the user question concisely based  on the context provided.\n"
    #     "ONLY output the direct answer as a word, number, address, or short phrase.\n"
    #     "Do NOT repeat the question or context. Do NOT give explanations or full sentences.\n\n"
    #     # f"User question: {data_dict['question']}\n\n"
    # )

    system_prompt = (
        "You are an expert on microcontrollers and can provide detailed information about their peripherals, registers, and fields.\n"
        "ONLY answer with valid register names, addresses, values or lists.\n"
        "Do NOT provide full sentence for the answer\n"
        "Do not explain. Do not repeat the question."
    )

    context = "\n- " + "\n- ".join(summaries) if summaries else ""

    prompt = (
        "[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
    )
    if few_shot_examples :
        # print(f"****few_shot_examples : {few_shot_examples}")
        prompt += f"{few_shot_examples}\n"
    # else :
    #     print("*****No few_shot examples")
    prompt += f"Now answer:\n"
    if context:
        prompt += f"Context:{context}\n"
    # else:
    #     print("*****No context retrived")

    prompt += f"Question: {question}\nAnswer: [/INST]"
    # Step 3: Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(curr_model.device)
    with torch.no_grad():
        outputs = curr_model.generate(
            **inputs,
            max_new_tokens=768,
            # temperature=0.7,  
            # top_p=0.95,
            # do_sample=True,
            do_sample=False,
            pad_token_id=model.config.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Step 4: Extract answer cleanly
    answer = generated_text.replace(prompt, "").strip()
    return answer.split("\n")[0].strip()
    # return answer

def load_chroma_db(local_directory):
    return Chroma(persist_directory=local_directory, embedding_function=embedding)


def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images, image_ocr_raw):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        # add_documents(retriever, image_summaries, images) #changing this as codellama doesn't deal directly with images
        add_documents(retriever, image_summaries, image_ocr_raw)
        # add_documents(retriever, image_summaries, image_summaries)

    return retriever

def init_rag(chroma_path, pickle_path):
    # if os.path.exists(db_path) and os.path.exists(pickle_path):
    print("Loading existing Chroma database...")
    vectorstore = load_chroma_db(chroma_path)
    
    with open(pickle_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Access the variables
    texts = loaded_data['texts']
    tables = loaded_data['tables']
    text_summaries = loaded_data['text_summaries']
    table_summaries = loaded_data['table_summaries']
    img_base64_list = loaded_data['img_base64_list']
    image_summaries = loaded_data['image_summaries']
    img_raw_ocr = loaded_data['img_raw_ocr']

    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
        img_raw_ocr,
    )
    return retriever_multi_vector_img


def compute_similarity(model_output: str, ground_truth: str) -> tuple:
    # Same as before
    embeddings_eval = eval_embedding.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings_eval[0]], [embeddings_eval[1]])[0][0]
    return cos_sim


def evaluate(name="", rag_retreiver = None, json_q_a_file_path="./nrf52840.json"):
    # Same as before
    # with open(json_q_a_file_path, 'r') as file:
    #     q_a = json.load(file)
    with open(json_q_a_file_path, 'r', encoding='utf-8') as file:
        q_a = [json.loads(line) for line in file]

    # Randomly select 3 examples for few-shot context
    few_shot_examples = random.sample(q_a, 3)
    few_shot_context=""

    # Create the few-shot context # i would just extract q & a it for openAI dataset
    few_shot_context = "\n".join(
        f"Example Question: {example['messages'][1]['content']}\n"
        f"Example Answer: {example['messages'][2]['content']}"
        for example in few_shot_examples
    ) + "\n\n"

    # for example in few_shot_examples :
    #     full_text = example["text"]
    #     if '[/INST]' in full_text:
    #         prompt_part, sep, answer_part = full_text.partition('[/INST]')
    #         question = prompt_part + sep  # keep [INST]...[/INST]
    #         ground_truth = answer_part.strip()
    #     else:
    #         # fallback if no [/INST] (just in case)
    #         question = full_text
    #         ground_truth = ""
    #     few_shot_context+=f"Example Question:{question}\nExample Answer:{ground_truth}\n"
    
    # few_shot_context+="\n\n"

    # print(f"******few_shot_context : {few_shot_context}")

    # Create a subset excluding the few-shot examples
    remaining_q_a = [example for example in q_a if example not in few_shot_examples]
    print(f"**** Len of remaining_q_a : {remaining_q_a}")
    scores = []

    def process_q_a(q_a_pair):
        # full_text = q_a_pair["text"]
        # question=""
        # ground_truth=""

        # if '[/INST]' in full_text:
        #     prompt_part, sep, answer_part = full_text.partition('[/INST]')
        #     question = prompt_part + sep  # keep [INST]...[/INST]
        #     ground_truth = answer_part.strip()
        # else:
        #     # fallback if no [/INST] (just in case)
        #     question = full_text
        #     ground_truth = ""
        question = q_a_pair["messages"][1]["content"]
        ground_truth = q_a_pair["messages"][2]["content"]
        print(f"Processing q_a_pair : question: {question}, ground_truth : {ground_truth} ")
        rag_model_output = ask_bot(rag_retreiver, question, model)
        few_shot_model_output = ask_bot(rag_retreiver, question, model, few_shot_context)
        ft_rag_model_output = ask_bot(rag_retreiver,question,ft_model)
        ft_few_shot_model_output = ask_bot(rag_retreiver,question,ft_model,few_shot_context)

        rag_cos_sim = compute_similarity(rag_model_output, ground_truth)
        few_shot_cos_sim = compute_similarity(few_shot_model_output, ground_truth)
        ft_reg_cos_sim = compute_similarity(ft_rag_model_output, ground_truth)
        ft_few_shot_cos_sim = compute_similarity(ft_few_shot_model_output, ground_truth)

        return {
            'datasheet': name,
            'question': question,
            'ground_truth': ground_truth,
            'baseline model_output': rag_model_output,
            'few shot model_output': few_shot_model_output,
            'ft model output' : ft_rag_model_output,
            'ft few_shot model output' : ft_few_shot_model_output,
            'baseline cosine_similarity': rag_cos_sim,
            'ft cosine_similarity': ft_reg_cos_sim,
            'few_shot cosine_similarity': few_shot_cos_sim,
            'ft few_shot cosine_similarity': ft_few_shot_cos_sim

        }

    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_q_a, q_a_pair) for q_a_pair in remaining_q_a]
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         scores.append(future.result())

    for q_a_pair in remaining_q_a :
        result = process_q_a(q_a_pair)
        scores.append(result)

    return scores

mcu_name = "nRF52840"
chroma_db_path = f"hf_codellama_7b_exp/chroma_dbs/{mcu_name}_db"
pickle_path = f"hf_codellama_7b_exp/pickle_files/{mcu_name}_summarized.pkl"
dataset_path = f"./evaluation_mcu_svd_dataset/datasets_{mcu_name}"

assert os.path.exists(chroma_db_path)
assert os.path.exists(pickle_path)
assert os.path.exists(dataset_path)

all_datasheet_scores = []
for key, file_paths in [(mcu_name, [chroma_db_path,pickle_path,dataset_path])]:
    print(key)
    chroma_db_path, pickle_path, eval_q_a_json_path = file_paths
    rag_retreiver = init_rag(chroma_db_path, pickle_path)
 
    print("Finished Creating RAG pipeline")

    main_data_path = os.path.join(eval_q_a_json_path, "main_data.jsonl")
    all_scores = evaluate(key, rag_retreiver, main_data_path)
    print("Finished Evaluation")

    all_datasheet_scores.extend(all_scores)


csv_file = f"hf_codellama_7b_exp/eval_outcome/{mcu_name}_output.csv"

# Define the column names based on dictionary keys
fieldnames = all_datasheet_scores[0].keys()

# Writing to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header
    writer.writerows(all_datasheet_scores)  # Write the data