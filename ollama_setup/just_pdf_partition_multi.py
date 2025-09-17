#This script is for partitioning a given pdf into text,images, tables; and store it in a pickle file that can later be retrived for embedding
from unstructured.partition.pdf import partition_pdf
import pytesseract
from tqdm import tqdm
import pickle

import os
import base64
import time
import multiprocessing
from multiprocessing import Process, cpu_count
from pathlib import Path
import sys

print("Imports Done!!")

# Directory with pdf files
PDF_DIR = "./pdfs"
RAW_DATA_COLLECT_DIR = "./pdf_raw_elements"
IMAGE_COLLECT_DIR = "./images_collect"

# pdf_path = "../pdfs/nRF52840_PS_v1.11.pdf"
# raw_data_path = "../pdf_raw_elements/nrf52840_raw.pkl"
# image_path = "../images_collect/images_nrf52820"

print("Partitioning PDF...")

#Parallelize it later
  
def process_file(pdf_path):
    path = Path(pdf_path)
    pdf_full_name = path.name           # e.g., 'nrf52840.pdf'
    mcu_name = path.stem           # e.g., 'nrf52840' (no .pdf)
    raw_data_path = f"{RAW_DATA_COLLECT_DIR}/{mcu_name}_raw.pkl"
    image_path = f"{IMAGE_COLLECT_DIR}/images_{mcu_name}"
    print(f"Processing pdf : {pdf_full_name},raw_data_path : {raw_data_path}, image_path : {image_path} ")
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

    with open(raw_data_path, 'wb') as f:
        pickle.dump(pdf_elements,f)

    print(f"Done : pdf : {pdf_path}, raw_pdf_elements : {raw_data_path}!!")

def main():
    input_dir = Path(PDF_DIR)
    if not os.path.isdir(input_dir) or not os.path.isdir(Path(RAW_DATA_COLLECT_DIR)) or not os.path.isdir(Path(IMAGE_COLLECT_DIR)):
        print(f"Error: Directory, either '{input_dir}' or '{RAW_DATA_COLLECT_DIR}' or '{IMAGE_COLLECT_DIR}' does not exist.")
        sys.exit(1)
    files = list(input_dir.glob("*.pdf"))  # all pdf files in the dir
    workers = min(len(files), os.cpu_count()//2 or 1)
    print(f"No. of workes : {workers}")
    with multiprocessing.Pool(workers) as pool :
        pool.map(process_file, map(str,files))

    # processes = []
    # for file_path in files:
    #     p = Process(target=process_file, args=(str(file_path),))
    #     p.start()
    #     processes.append(p)

    # # Wait for all processes to finish
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    main()
    print("All Done!!")

