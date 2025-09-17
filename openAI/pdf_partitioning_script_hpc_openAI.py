#This script is for partitioning a given pdf into summaries text,images, tables; and store it in chromadb vector store, and pickle file that we store locally and directly use later to save time.
# It uses openAI with lanchain; chain embedding model depending on model that you plan to use.
from unstructured.partition.pdf import partition_pdf
import pytesseract
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
import os
import base64
import time

load_dotenv()
print("Imports Done!!")

pdf_path = "../data/pdfs/nRF52820_PS_v1.4.pdf"
image_path = "../data/images_collect/images_nrf52820"
db_path = "../data/pdf_partitioning_result/chroma_dbs/chroma_db_nrf52820"
pickle_path = "../data/pdf_partitioning_result/pickle_files/nrf52820.pkl"

def load_chroma_db(local_directory=db_path):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=local_directory, embedding_function=embeddings)

def categorize_elements(raw_pdf_elements):
    text_elements = []
    table_elements = []
    for element in raw_pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text_elements.append(str(element))
        elif 'Table' in str(type(element)):
            table_elements.append(str(element))
    return text_elements, table_elements

model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024)

print("OpenAI model initiated")

# Generate summaries of text elements
def generate_text_summaries(texts, tables, summarize_texts=False):
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
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Make image summary"""
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """
    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Include all the values in each image, including extracting all the text. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries

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


if os.path.exists(db_path) and os.path.exists(pickle_path):
    print("Loading existing Chroma database...")
    vectorstore = load_chroma_db()

    with open(pickle_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Access the variables
    texts = loaded_data['texts']
    tables = loaded_data['tables']
    text_summaries = loaded_data['text_summaries']
    table_summaries = loaded_data['table_summaries']
    img_base64_list = loaded_data['img_base64_list']
    image_summaries = loaded_data['image_summaries']

else:
    print("Creating new Chroma database...")
    # Store embeddings in Chroma
    start_time = time.time()
    pdf_elements = partition_pdf(
        pdf_path,
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        extract_image_block_types=['Table', 'Image'],
        extract_image_block_output_dir=image_path,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_path
    )
    end_time = time.time()
    print(f"pdf partition done!! Time : {end_time-start_time}")
    
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
    img_base64_list, image_summaries = generate_img_summaries(image_path)
    end_time = time.time()
    print(f"generate img summaries done!! Time : {end_time - start_time}")

    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'texts': texts,
            'tables': tables,
            'text_summaries': text_summaries,
            'table_summaries': table_summaries,
            'img_base64_list': img_base64_list,
            'image_summaries': image_summaries
        }, f)
    print("Dumped pickle file")

    start_time = time.time()
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        persist_directory=db_path
    )
    end_time = time.time()
    print(f"vectorstore done!! Time : {end_time - start_time}")

print("Done!!")
