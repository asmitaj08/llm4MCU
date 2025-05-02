print("Helloo")
#This script is for partitioning a given pdf into text,images, tables; and store it in a pickle file that can later be retrived for embedding
from unstructured.partition.pdf import partition_pdf
import pytesseract
from tqdm import tqdm
import pickle

import os
import base64
import time

print("Imports Done!!")

pdf_path = "../pdfs/nRF52840_PS_v1.11.pdf"
raw_data_path = "../pdf_raw_elements/nrf52840_raw.pkl"
image_path = "../images_collect/images_nrf52820"

print("Partitioning PDF...")

#Parallelize it later
  
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
