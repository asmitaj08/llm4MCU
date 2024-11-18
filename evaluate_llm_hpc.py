from unstructured.partition.pdf import partition_pdf
# import pytesseract
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import pickle
from collections import defaultdict
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import openai
from openai import OpenAI
import csv
from IPython.display import HTML, display
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from PIL import Image
import io
import re

load_dotenv()

chroma_dir = "./pdf_partitioning_result/chroma_dbs"
pickle_dir = "./pdf_partitioning_result/pickle_files"
evaluate_dir = "./evaluation_mcu_svd_dataset"

dataset_dict = defaultdict(list)

def extract_key(filename):
    return filename.split('db_')[-1] if '_' in filename else filename.split('.pkl')[0]

def qa_key(filename):
    return filename.split('datasets_')[-1]

# Aggregate file paths
for root, dirs, files in os.walk(chroma_dir):
    for dirname in dirs:
        key = extract_key(dirname)
        # print(key)
        # print('lpc1102_04.pkl')
        dataset_dict[key].append(os.path.join(root, dirname))

for root, dirs, files in os.walk(pickle_dir):
    for file in files:
        # print(file.split('.pkl')[0])
        key = extract_key(file)
        if key == 'lpc1102_04.pkl':
            key = 'lpc1102_04'
        dataset_dict[key].append(os.path.join(root, file))

for root, dirs, files in os.walk(evaluate_dir):
    for dirname in dirs:
        key = qa_key(dirname)
        # print(key)
        if key == 'qn908xc':
            key = 'QN9080x'
        elif key == 'stm32f100xx':
            key = 'stm32f100'
        dataset_dict[key].append(os.path.join(root, dirname))


dataset_dict = dict(dataset_dict)
#print(f"dataset_dict : {dataset_dict}")

def load_chroma_db(local_directory):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=local_directory, embedding_function=embeddings)

model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024)

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
        add_documents(retriever, image_summaries, images)

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
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    if len(b64_images) > 0:
        return {"images": b64_images[:1], "texts": []}
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    messages = []

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are an AI scientist tasking with providing factual answers from a datasheet of a System-on-Chip (SoC) \n"
            "Use this information to provide answers related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
        ),
    }
    messages.append(text_message)
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model  # MM_LLM
        | StrOutputParser()
    )

    return chain

def ask_bot(chain_multimodal_rag, query):
    # docs = retriever_multi_vector_img.get_relevant_documents(query, limit=10)
    # print(split_image_text_types(docs))
    return chain_multimodal_rag.invoke(query)

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

# Load your finetuned LLM model here
# For example:
# model = YourLLMModel.load_from_checkpoint('path_to_checkpoint')

# Load the pre-trained Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(model_output: str, ground_truth: str) -> tuple:
    # Same as before
    embeddings = embedding_model.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def evaluate(name="", rag = None, json_q_a_file_path="./nrf52840.json"):
    # Same as before
    # with open(json_q_a_file_path, 'r') as file:
    #     q_a = json.load(file)
    with open(json_q_a_file_path, 'r', encoding='utf-8') as file:
        q_a = [json.loads(line) for line in file]

    # Randomly select 3 examples for few-shot context
    few_shot_examples = random.sample(q_a, 3)

    # Create the few-shot context
    few_shot_context = "\n".join(
        f"Example Question: {example['messages'][0]['content']} {example['messages'][1]['content']}\n"
        f"Example Answer: {example['messages'][2]['content']}"
        for example in few_shot_examples
    ) + "\n\n"

    # Create a subset excluding the few-shot examples
    remaining_q_a = [example for example in q_a if example not in few_shot_examples]


    scores = []

    def process_q_a(q_a_pair):
        question = q_a_pair["messages"][0]["content"] + " " + q_a_pair["messages"][1]["content"]
        ground_truth = q_a_pair["messages"][2]["content"]
        reg_model_output = ask_bot(rag, question)
        few_shot_model_output = ask_bot(rag, few_shot_context + " " + question)

        reg_cos_sim = compute_similarity(reg_model_output, ground_truth)
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

for key, file_paths in [('stm32f0xx', ['./pdf_partitioning_result/chroma_dbs/chroma_db_stm32f0xx',
 './pdf_partitioning_result/pickle_files/stm32f0xx.pkl',
 './evaluation_mcu_svd_dataset/datasets_stm32f0xx'])]:
    print(key)
    chroma_db_path, pickle_path, eval_q_a_json_path = file_paths
    rag_pipeline = init_rag(chroma_db_path, pickle_path)
    print("Finished Creating RAG pipeline")

    main_data_path = os.path.join(eval_q_a_json_path, "main_data.jsonl")
    all_scores = evaluate(key, rag_pipeline, main_data_path)
    print("Finished Evaluation")

    all_datasheet_scores.extend(all_scores)


csv_file = 'stm32f0xx_output.csv'

# Define the column names based on dictionary keys
fieldnames = all_datasheet_scores[0].keys()

# Writing to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header
    writer.writerows(all_datasheet_scores)  # Write the data
