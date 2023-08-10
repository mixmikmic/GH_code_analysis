import os 
import urllib
from os import path
import json
import requests
import time

# Check that docker is working
get_ipython().system('docker run --rm hello-world')

get_ipython().run_cell_magic('bash', '', 'mkdir script\nmkdir script/code')

get_ipython().run_cell_magic('writefile', 'script/code/model.py', '\nimport base64\nimport urllib\nimport numpy as np\nimport cntk\nimport pkg_resources\nfrom flask import Flask, json, request\nfrom io import BytesIO\nfrom PIL import Image, ImageOps\nfrom cntk import load_model, combine\n\napp = Flask(__name__)\nprint("Something outside of @app.route() is always loaded")\n\n# Pre-load model\nMODEL = load_model("ResNet_18.model")\nprint("Loaded model: ", MODEL)\n# Pre-load labels\nwith open(\'synset-1k.txt\', \'r\') as f:\n    LABELS = [l.rstrip() for l in f]\nprint("Loaded {0} labels".format(len(LABELS)))\n\n@app.route("/")\ndef healthy_me():\n    return "healthy"\n\n@app.route(\'/cntk\')\ndef cntk_ver():\n    return "CNTK version: {}".format(pkg_resources.get_distribution("cntk").version)\n\n@app.route(\'/posttest\', methods=[\'POST\'])\ndef posttest():\n    return "POST healthy"\n\n@app.route("/api/uploader", methods=[\'POST\'])\ndef api_upload_file():\n    img = Image.open(BytesIO(request.files[\'imagefile\'].read())).convert(\'RGB\')\n    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)\n    return json.dumps(run_some_deep_learning_cntk(img))\n\ndef run_some_deep_learning_cntk(rgb_pil_image):\n    # Convert to BGR\n    rgb_image = np.array(rgb_pil_image, dtype=np.float32)\n    bgr_image = rgb_image[..., [2, 1, 0]]\n    img = np.ascontiguousarray(np.rollaxis(bgr_image, 2))\n\n    # Use last layer to make prediction\n    z_out = combine([MODEL.outputs[3].owner])\n    result = np.squeeze(z_out.eval({z_out.arguments[0]: [img]}))\n\n    # Sort probabilities \n    a = np.argsort(result)[-1]\n    predicted_category = " ".join(LABELS[a].split(" ")[1:])\n    \n    return predicted_category\n\nif __name__ == \'__main__\':\n    # This is just for debugging\n    app.run(host=\'0.0.0.0\', port=5005)')

get_ipython().run_cell_magic('writefile', 'script/code/requirements.txt', 'Flask\ngunicorn\npillow')

urllib.urlretrieve('https://github.com/ilkarman/Azure-WebApp-w-CNTK/raw/master/Model/ResNet_18.model', 'script/code/ResNet_18.model')

urllib.urlretrieve('https://github.com/ilkarman/Azure-WebApp-w-CNTK/raw/master/Model/synset-1k.txt', 'script/code/synset-1k.txt')

get_ipython().system('az login -o table')

selected_subscription = "'.....'"

get_ipython().system('az account set --subscription $selected_subscription')

docker_registry = "ikmscontainer"
docker_registry_group = "ikmscontainergorup"

get_ipython().system('az group create -n $docker_registry_group -l southcentralus -o table')

get_ipython().system('az acr create -n $docker_registry -g $docker_registry_group -l southcentralus -o table')

get_ipython().system('az acr update -n $docker_registry --admin-enabled true -o table')

json_data = get_ipython().getoutput('az acr credential show -n $docker_registry')
docker_username = json.loads(''.join(json_data))['username']
docker_password = json.loads(''.join(json_data))['password']

print(docker_username)
print(docker_password)

json_data = get_ipython().getoutput('az acr show -n $docker_registry')
docker_registry_server = json.loads(''.join(json_data))['loginServer']

get_ipython().system('mkdir script/docker')

# Using CNTK docker doesn't work - too big for WebApp
"""
%%writefile script/docker/dockerfile

FROM microsoft/cntk:2.0.beta15.0-cpu-python3.5
MAINTAINER Ilia Karmanov
ADD code /code
ENV PATH /root/anaconda3/envs/cntk-py35/bin:$PATH
WORKDIR /code
RUN pip install -r requirements.txt && \
    sudo rm -R /cntk/Examples && \
    sudo rm -R /cntk/Tutorials 

EXPOSE 5005
CMD ["python", "model.py"]
"""

get_ipython().run_cell_magic('writefile', 'script/docker/dockerfile', '\nFROM python:3.5-slim\nMAINTAINER Ilia Karmanov\nADD code /code\nENV PATH /usr/local/mpi/bin:$PATH\nENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH\nWORKDIR /code\nRUN apt-get update \\\n    && apt-get install -y --no-install-recommends wget build-essential \\\n    && rm -rf /var/lib/apt/lists/* \\\n    && wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz \\\n    && tar -xzvf ./openmpi-1.10.3.tar.gz \\\n    && cd openmpi-1.10.3 \\\n    && ./configure --prefix=/usr/local/mpi \\\n    && make -j all \\\n    && make install \\\n    && cd .. \\\n    && rm -R openmpi-1.10.3 \\\n    && rm openmpi-1.10.3.tar.gz \\\n    && pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.0.beta15.0-cp35-cp35m-linux_x86_64.whl \\\n    && pip install -r requirements.txt \\\n    && apt-get purge -y --auto-remove wget build-essential\n\nEXPOSE 5005\nCMD ["/usr/local/bin/gunicorn", "--bind", "0.0.0.0:5005", "model:app"]')

container_name = docker_registry_server + "/ilkarman/dockergunicorn"
application_path = 'script'
docker_file_location = path.join(application_path, 'docker/dockerfile')

get_ipython().system('docker login $docker_registry_server -u $docker_username -p $docker_password')

get_ipython().run_cell_magic('bash', '', 'docker stop $(docker ps -a -q)\ndocker rm $(docker ps -a -q)')

# Running from shell:
docker_build = "docker build -t {0} -f {1} {2} --no-cache".format(container_name, docker_file_location, application_path)
docker_build

# This will take a while; potentially run from shell instead to see output (there will be a lot)
build_out = get_ipython().getoutput('$docker_build')

# 1.23GB (ResNet_18.model is ~60MB)
#!docker images   

# To debug
print(container_name)
# In shell (run interactive mode):
#docker run -it $container_name /bin/bash
#conda info --env
#which python
# ... etc

test_cont = get_ipython().getoutput('docker run -p 5005:5005 -d $container_name')

time.sleep(5)  # Wait to load
get_ipython().system('curl http://0.0.0.0:5005')

get_ipython().system('curl http://0.0.0.0:5005/cntk')

requests.post("http://0.0.0.0:5005/posttest").content

hippo_url = "https://i.ytimg.com/vi/96xC5JIkIpQ/maxresdefault.jpg"
fname = urllib.urlretrieve(hippo_url, "bhippo.jpg")[0]
requests.post("http://0.0.0.0:5005/api/uploader", files={'imagefile': open(fname, 'rb')}).json()

get_ipython().system('docker kill {test_cont[0]}')

get_ipython().system('docker push $container_name')

get_ipython().system('curl http://ikdockergunicorn.azurewebsites.net/')

get_ipython().system('curl http://ikdockergunicorn.azurewebsites.net/cntk')

requests.post("http://ikdockergunicorn.azurewebsites.net/posttest").content

requests.post("http://ikdockergunicorn.azurewebsites.net/uploader", files={'imagefile': open(fname, 'rb')}).content



