## Ollama setup on hpc 
 
*  ```curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
     mkdir ollama
     tar -C ollama -xzf ollama-linux-amd64.tgz
     ./ollama/bin/ollama serve```

* run "`./ollama/bin/ollama serve`" within a srun/sbatch job on the node you want the processing to happen on

#### ollam_serve job script :
```
    #!/bin/bash -l 
    #SBATCH -J ollama_serv
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --time=48:00:00
    #SBATCH --mem=10G
    #SBATCH -o ollama_serv-%j.output
    #SBATCH -e ollama_serv-%j.output
    #SBATCH --account=xxxxxx
    #SBATCH --partition=gpu-xxxxx
    #SBATCH --gres=gpu:1

source /software/conda3/4.X/etc/profile.d/conda.sh && \
	conda activate llm_env && \
	/home/user/ollama/bin/ollama serve
```

#### running jupyter notebook, that can be accessed on  local machine using ssh tunneling 

```
#!/bin/bash -l 
#SBATCH -J jupyter_nb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem=40G
#SBATCH -o jupyter_nb-%j.output
#SBATCH -e jupyter_nb-%j.output
#SBATCH --account=xxxxx 
#SBATCH --partition=gpu-xxxxx 
#SBATCH --gres=gpu:1

source /software/conda3/4.X/etc/profile.d/conda.sh && \
	conda activate llm_env && \
	#/home/user/ollama/bin/ollama serve
    # python3 -u /home/user/pdf_script.py > log_nrf52840.log 2>&1
    jupyter notebook --no-browser  --port=xxxx --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --notebook-dir="/home/user/path/"
```

* Then, in my local machine,  `ssh -L xxxx:localhost:xxxx -J user@hpc.xx.xx user@hpc-node` , now you can directly jump to your node in the hpc cluster, and on local host, you can access jupyter notebook using localhost:xxxx (it will ask you for token, then check `jupyter_nb-%j.output` where it gets logged)

### Running Inside Jupyter Notebook
* Ollama has python APIs that you can use.
* And some commands are : `!/home/xxx/ollama/bin/ollama list`, `!/home/xxx/ollama/bin/ollama pull model_name(check their github/doc)`, `!/home/xxx/ollama/bin/ollama run model_name`, `!/home/xxx/ollama/bin/ollama ps` 
`!pip install ollama`, `!/home/xxx/ollama/bin/ollama stop llama3.2:latest`

* Sample Script :
    ```
        from ollama import chat
        from ollama import ChatResponse

        response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
        ])
        print(response['message']['content'])
        # or access fields directly from the response object
        print(response.message.content)
    ```
* Stop using : `!/home/xxx/ollama/bin/ollama stop llama3.2:latest`


