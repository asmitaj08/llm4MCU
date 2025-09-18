# datasheet-llm

N.B : Used Python Version : 3.10

- Use conda environment to do the required installation. 
- The required package installation can be found in `requirements.txt`, and the first cell (pip install ..) in *.ipynb files in respective `openAI` , `huggingFace_exp` folders.
- All experiments performed with GPT-4o model are provided in `openAI` folder, experiments with codellama using hugging face is provided in `huggingFace_exp/hf_codellama_7b_exp` dir
- We also experimented with ollama as provided in folder `ollama_setup`
<!-- After cloning this repo, add the env file to the directory (With openAI key)

Run the first cell (pip...) to install all necessary libraries 

Works best in a conda environment 

use uart.pdf for quick testing (estimate ~2 hours to ingest the pdf on CPU),  need to check how to run this using gpu, preferably within generate_text_summaries and generate_image_summaries and encode image 

a vectore store is in the repo, I need to still figure out how to use it instead of ingesting the pdf every time I run the notebook.  -->

