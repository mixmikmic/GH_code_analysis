get_ipython().run_cell_magic('bash', '', "# create config file for jupyter (file copied by Dockerfile)\ncat << LINES > jupyter_notebook_config.py\nc.NotebookApp.ip = '*'\nc.NotebookApp.port = 8888\nc.NotebookApp.token = ''\nc.NotebookApp.password = ''\nLINES")

get_ipython().system('ls')

get_ipython().run_cell_magic('bash', '', 'cat << LINES > Dockerfile\n# replace the base image name\nFROM nvcr.io/nvidia/tensorflow:17.12\n\n# install necessary packages\nRUN apt-get update && apt-get install -y \\\n        libzmq3-dev \\\n        python-dev \\\n        python-matplotlib \\\n        python-pandas \\\n        python-pip \\\n        python-sklearn && \\\n    rm -rf /var/lib/apt/lists/*\n\n# \nRUN pip install \\\n        ipykernel \\\n        jupyter && \\\n    python -m ipykernel.kernelspec\n\nCOPY jupyter_notebook_config.py /root/.jupyter/\n\n\nWORKDIR /workspace\nVOLUME /workspace\n\nEXPOSE 8888\n\nCMD ["jupyter", "notebook", "--allow-root"]\n\nLINES')

