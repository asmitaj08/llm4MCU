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
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm

load_dotenv()
base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
# embedd_model = "sentence-transformers/all-MiniLM-L6-v2" #will have to experiment with embeddig models
# embedd_model = "intfloat/e5-large-v2"
embedd_model = "BAAI/bge-large-en-v1.5"

pdf_path = "pdfs/nRF52840.pdf"
image_path = "images_collect/images_nRF52840/"
pdf_raw_data_path = "pdf_raw_elements_common/nRF52840_raw.pkl"
db_path = "hf_codellama_7b_exp/chroma_dbs/nRF52840_db_parallel"
summarized_pickle_path = "hf_codellama_7b_exp/pickle_files/nRF52840_summarized_parallel.pkl"

embedding = HuggingFaceEmbeddings(model_name=embedd_model)

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

def categorize_elements(raw_pdf_elements):
    text_elements = []
    table_elements = []
    for element in raw_pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text_elements.append(str(element))
        elif 'Table' in str(type(element)):
            table_elements.append(str(element))
    return text_elements, table_elements

def generate_text_summaries(texts, tables, summarize_texts=False, batch_size=32, concurrency=4):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well-optimized for retrieval. \
    Don't use Markdown, just plain text output. Table \
    or text: {element} """
    prompt = PromptTemplate.from_template(prompt_text)

    # Text summary chain
    summarize_chain = {"element": lambda x: x} | prompt | model_pipe | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    # if texts and summarize_texts:
    #     text_summaries = summarize_chain.batch(texts, {"max_concurrency": 4})
    # elif texts:
    #     text_summaries = texts
    # Text
    if texts and summarize_texts:
        print(f"Summarizing {len(texts)} texts in batches of {batch_size} with concurrency={concurrency}")
        for i in tqdm(range(0, len(texts), batch_size), desc="Text summary batches"):
            chunk = texts[i:i + batch_size]
            out = summarize_chain.batch(chunk, {"max_concurrency": concurrency})
            text_summaries.extend(out)
    else:
        text_summaries = texts or []

    # Apply to tables if tables are provided
    # if tables:
    #     table_summaries = summarize_chain.batch(tables, {"max_concurrency": 4})

    # Tables
    if tables:
        print(f"Summarizing {len(tables)} tables in batches of {batch_size} with concurrency={concurrency}")
        for i in tqdm(range(0, len(tables), batch_size), desc="Table summary batches"):
            chunk = tables[i:i + batch_size]
            out = summarize_chain.batch(chunk, {"max_concurrency": concurrency})
            table_summaries.extend(out)

    return text_summaries, table_summaries

# Load BLIP once (reuse across calls)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to("cuda")

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize_ocr(img_base64, prompt=None):
    """OCR-based image summarization"""
    # Decode base64 string to image
    img_data = base64.b64decode(img_base64)
    # image = Image.open(BytesIO(img_data)).convert("RGB")
    image = Image.open(BytesIO(img_data))
    # Preprocessing image
    image = image.convert("L")  # grayscale
    image = image.point(lambda x: 0 if x < 140 else 255)  # binarize

    # Extract text from image using pytesseract
    extracted_text = pytesseract.image_to_string(image)

    # Combine with prompt if needed
    if prompt:
        return f"{prompt.strip()}\n\nExtracted Text:\n{extracted_text.strip()}"
    else:
        return extracted_text.strip()

def ocr_then_blip(img_base64, prompt=None, min_ocr_chars=10):
    """Try OCR first; fallback to BLIP if OCR is too short"""

    # Decode base64 to image
    img_data = base64.b64decode(img_base64)
    image = Image.open(BytesIO(img_data))

    # Preprocessing: grayscale + binarize
    processed_image = image.convert("L")
    processed_image = processed_image.point(lambda x: 0 if x < 140 else 255)

    # Try OCR
    ocr_text = pytesseract.image_to_string(processed_image).strip()

    if len(ocr_text) >= min_ocr_chars:
        return f"{prompt.strip() if prompt else ''}\n\nExtracted Text:\n{ocr_text}"
    else:
        # Fallback to BLIP
        inputs = blip_processor(image.convert("RGB"), return_tensors="pt").to("cuda")
        out_ids = blip_model.generate(**inputs, max_new_tokens=64)
        blip_caption = blip_processor.decode(out_ids[0], skip_special_tokens=True)

        return f"{prompt.strip() if prompt else ''}\n\nBLIP Caption:\n{blip_caption}"

def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images using OCR
    path: Path to list of .jpg/.png files extracted by Unstructured
    """
    img_base64_list = []
    image_summaries = []

    prompt = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Include all the values in each image, including extracting all the text.
Give a concise summary of the image that is well optimized for retrieval."""

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)

            # summary = image_summarize_ocr(base64_image, prompt=prompt)
            summary = ocr_then_blip(base64_image, prompt=prompt)
            image_summaries.append(summary)

    return img_base64_list, image_summaries

def summarizer_worker(args):
        b64, p = args
        return ocr_then_blip(b64, prompt=p)

def generate_img_summaries_parallel(path, prompt=None, max_workers=None):
    """
    Parallel image summarization using OCR + BLIP fallback.
    Args:
        path (str): Directory path containing image files
        prompt (str): Optional prompt to pass to summarizer
        max_workers (int): Max processes to use (default: min(cpu_count(), 8))
    Returns:
        Tuple[List[base64_str], List[summary_str]]
    """
    prompt = prompt or """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Include all the values in each image, including extracting all the text.
Give a concise summary of the image that is well optimized for retrieval."""

    # Collect image files
    image_files = sorted([
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Encode images to base64
    base64_images = [encode_image(p) for p in tqdm(image_files, desc="Encoding images")]

    # Wrap prompt with base64 for each
    args = [(b64, prompt) for b64 in base64_images]

    # Parallel summarization
    max_workers = max_workers or min(cpu_count(), 8)
    # with Pool(processes=max_workers) as pool:
    with get_context("spawn").Pool(processes=max_workers) as pool:
        image_summaries = list(tqdm(pool.map(summarizer_worker, args), total=len(args), desc="Summarizing images"))

    return base64_images, image_summaries


if os.path.exists(pdf_raw_data_path):
    print(f"Loading existing raw pdf elements from {pdf_raw_data_path}...")
    with open(pdf_raw_data_path, 'rb') as f:
        pdf_elements = pickle.load(f)
   # extract tables and texts
    start_time = time.time()
    texts, tables = categorize_elements(pdf_elements)
    end_time = time.time()
    print(f"categorize elements done!! Time : {end_time - start_time}")

    # Get text & table summaries
    start_time = time.time()
    text_summaries, table_summaries = generate_text_summaries(texts[0:19], tables, summarize_texts=True)
    end_time = time.time()
    print(f"generate text summaries done!! Time : {end_time - start_time}")

    # Image summaries
    start_time = time.time()
    img_base64_list, image_summaries = generate_img_summaries_parallel(path=image_path, max_workers=12)
    end_time = time.time()
    print(f"generate img summaries done!! Time : {end_time - start_time}")

    with open(summarized_pickle_path, 'wb') as f:
        pickle.dump({
            'texts': texts,
            'tables': tables,
            'text_summaries': text_summaries,
            'table_summaries': table_summaries,
            'img_base64_list': img_base64_list,
            'image_summaries': image_summaries
        }, f)
    print(f"Dumped pickle file with summaries : {summarized_pickle_path} ")

    start_time = time.time()
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function =embedding,
        persist_directory=db_path
    )
    end_time = time.time()
    print(f"vectorstore done!! Time : {end_time - start_time}")


else:
    print(f"Error , exiting : raw pdf elements Not found : {pdf_raw_data_path}...")
    sys.exit(1)

print(f"Done : pdf : {pdf_path}, summarized_pickle_file : {summarized_pickle_path}, chroma_db : {db_path}!!")



