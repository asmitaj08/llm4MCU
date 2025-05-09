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

chroma_db_path = "hf_codellama_7b_exp/chroma_dbs/nRF52840_db"
pickle_path = "hf_codellama_7b_exp/pickle_files/nRF52840_summarized.pkl"

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

# Create Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=model.config.eos_token_id, #avoiding warning Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    max_new_tokens=768,
    # temperature=0.0, # no need as do_sample=False
    do_sample=False
)

# Wrap it in LangChain-compatible interface
model_pipe = HuggingFacePipeline(pipeline=pipe)

##ft 
ft_model_dir = "codellama_7b/final_model"
assert os.path.exists(ft_model_dir), f"Finetuned Model directory not found: {ft_model_dir}"

# Load LoRA adapter
ft_model = PeftModel.from_pretrained(model, ft_model_dir)
ft_model.eval()

def ask_bot_ft(question: str) -> str:
    formatted_prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(ft_model.device)
    with torch.no_grad():
        outputs = ft_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.replace(formatted_prompt, "").strip()
    # Optionally post-process
    # return generated_text.replace(formatted_prompt, "").strip()
    # Normalize: clean each line
    lines = [
        re.sub(r"[.,:;!?]+$", "", line.strip())  # Remove trailing punctuation
        for line in answer.splitlines()
        if line.strip()
    ]

    return ", ".join(lines)

def load_chroma_db(local_directory=chroma_db_path):
    return Chroma(persist_directory=local_directory, embedding_function=embedding)


def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
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
        add_documents(retriever, image_summaries, images) #changing this as codellama doesn't deal directly with images
        add_documents(retriever, image_summaries, image_summaries)

    return retriever

def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            content = doc.page_content
        else:
            content = doc

        # If this document is a base64 image (raw), skip it
        if looks_like_base64(content) and is_image_data(content): #we don'rt need it in this case as we r not dealing with images during inferences
            # print("\n**********found base64")
            continue  # raw image, not usable here
        else:
            texts.append(content)

    return {"images": [], "texts": texts}


def text_only_prompt_func(data_dict):
    """
    Formats a CodeLLaMA-compatible prompt with summaries and question.
    """
    prompt = (
        "You are an expert on microcontrollers and can provide detailed information about their peripherals, registers, and fields.\n"
        "Answer the user question concisely based  on the context provided.\n"
        "ONLY output the direct answer as a word, number, address, or short phrase.\n"
        "Do NOT repeat the question or context. Do NOT give explanations or full sentences.\n\n"
        # f"User question: {data_dict['question']}\n\n"
    )
    if data_dict["context"]["texts"]:
        prompt += "Relevant Context:\n" + "\n\n".join(data_dict["context"]["texts"])

    instruction_block = (
        f"Context:\n{prompt}\n\n"
        f"Question: {data_dict['question']}"
    )

    # Final CodeLLaMA prompt format
    return f"### Instruction:\n{instruction_block}\n\n### Response:\n"

def multi_modal_rag_chain(retriever, curr_model):
    """
    Multi-modal RAG chain
    """

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(text_only_prompt_func)
        | curr_model  # MM_LLM
        | StrOutputParser()
    )

    return chain

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

    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    return chain_multimodal_rag

rag_pipeline = init_rag(chroma_db_path, pickle_path)

def ask_bot(chain_multimodal_rag, query):
    raw_output = chain_multimodal_rag.invoke(query)
    # print(f"Raw output : {raw_output}\n")
    # print("*********************")

    # Extract after prompt marker
    if "### Response:" in raw_output:
        answer = raw_output.split("### Response:")[-1].strip()
    else:
        answer = raw_output.strip()

    # Normalize: clean each line
    lines = [
        re.sub(r"[.,:;!?]+$", "", line.strip())  # Remove trailing punctuation
        for line in answer.splitlines()
        if line.strip()
    ]

    return ", ".join(lines)

def compute_similarity(model_output: str, ground_truth: str) -> tuple:
    # Same as before
    embeddings_eval = eval_embedding.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings_eval[0]], [embeddings_eval[1]])[0][0]
    return cos_sim


def evaluate(name="", rag = None, json_q_a_file_path="./nrf52840.json"):
    # Same as before
    # with open(json_q_a_file_path, 'r') as file:
    #     q_a = json.load(file)
    with open(json_q_a_file_path, 'r', encoding='utf-8') as file:
        q_a = [json.loads(line) for line in file]

    # Randomly select 3 examples for few-shot context
    few_shot_examples = random.sample(q_a, 3)
    few_shot_context=""

    # # Create the few-shot context
    # few_shot_context = "\n".join(
    #     f"Example Question: {example['messages'][0]['content']} {example['messages'][1]['content']}\n"
    #     f"Example Answer: {example['messages'][2]['content']}"
    #     for example in few_shot_examples
    # ) + "\n\n"

    for example in few_shot_examples :
        full_text = example["text"]
        if '[/INST]' in full_text:
            prompt_part, sep, answer_part = full_text.partition('[/INST]')
            question = prompt_part + sep  # keep [INST]...[/INST]
            ground_truth = answer_part.strip()
        else:
            # fallback if no [/INST] (just in case)
            question = full_text
            ground_truth = ""
        few_shot_context+=f"Example Question:{question}\nExample Answer:{ground_truth}\n"
    
    few_shot_context+="\n\n"

    print(f"******few_shot_context : {few_shot_context}")

    # Create a subset excluding the few-shot examples
    remaining_q_a = [example for example in q_a if example not in few_shot_examples]
    scores = []

    def process_q_a(q_a_pair):
        full_text = q_a_pair["text"]
        question=""
        ground_truth=""

        if '[/INST]' in full_text:
            prompt_part, sep, answer_part = full_text.partition('[/INST]')
            question = prompt_part + sep  # keep [INST]...[/INST]
            ground_truth = answer_part.strip()
        else:
            # fallback if no [/INST] (just in case)
            question = full_text
            ground_truth = ""
      
        rag_model_output = ask_bot(rag, question)
        few_shot_model_output = ask_bot(rag, few_shot_context + " " + question)

        rag_cos_sim = compute_similarity(rag_model_output, ground_truth)
        few_shot_cos_sim = compute_similarity(few_shot_model_output, ground_truth)


        ft_reg = client.chat.completions.create(
            model='ft:gpt-4o-2024-08-06:ucd-aseec:svd-finetune:AUNjCF3m',
            messages=[q_a_pair["messages"][0], q_a_pair["messages"][1]],
            max_tokens=1024,
            temperature=0
        )
        ft_reg_model_output = ft_reg.choices[0].message.content

        ft_few_shot = client.chat.completions.create(
            model='ft:gpt-4o-2024-08-06:ucd-aseec:svd-finetune:AUNjCF3m',
            messages=[{"role": "system", "content":few_shot_context}, q_a_pair["messages"][0], q_a_pair["messages"][1]],
            max_tokens=1024,
            temperature=0
        )

        ft_few_shot_model_output = ft_few_shot.choices[0].message.content

        ft_reg_cos_sim = compute_similarity(ft_reg_model_output, ground_truth)
        ft_few_shot_cos_sim = compute_similarity(ft_few_shot_model_output, ground_truth)

        return {
            'datasheet': name,
            'question': question,
            'ground_truth': ground_truth,
            'baseline model_output': reg_model_output,
            'few shot model_output': few_shot_model_output,
            'baseline cosine_similarity': reg_cos_sim,
            'ft cosine_similarity': ft_reg_cos_sim,
            'few_shot cosine_similarity': few_shot_cos_sim,
            'ft few_shot cosine_similarity': ft_few_shot_cos_sim

        }

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_q_a, q_a_pair) for q_a_pair in remaining_q_a]
        for future in tqdm(as_completed(futures), total=len(futures)):
            scores.append(future.result())

    return scores