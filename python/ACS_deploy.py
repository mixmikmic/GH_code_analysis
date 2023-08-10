get_ipython().system('pip install azuremlcli asyncio aiohttp')

get_ipython().system('pip install azure-cli -I --upgrade')

# Creating ssh key pair and saving it in the .library for re-use between containers
import os
if not os.path.exists('/home/nbuser/.ssh/id_rsa'):
    get_ipython().system('ssh-keygen -t rsa -b 2048 -N "" -f ~/.ssh/id_rsa')
print('Private key id_rsa:')
get_ipython().system('cat ~/.ssh/id_rsa')
print('Public key id_rsa.pub:')
get_ipython().system('cat ~/.ssh/id_rsa.pub')

get_ipython().system('az login -o table')

subscription = "<YOUR_SUBSCRIPTION_NAME_HERE>"
subscription = "'" + subscription + "'"
get_ipython().system('az account set --subscription $subscription')

import uuid

name = "aiimmersion{}".format(str(uuid.uuid4())[:8])

# Creating the environment
get_ipython().system('aml env setup --name $name')

ACS_deployment_key = "<YOUR_ACS_DEPLOYMENT_KEY>"

if "YOUR_ACS_DEPLOYMENT_KEY" in ACS_deployment_key:
    print("/!\ STOP /!\ You need to modify the value of ACS_deployment_key, please follow the above instructions")
else:
    print("You are good to go :)")

get_ipython().system('aml env setup -s $ACS_deployment_key')

get_ipython().system('cat ~/.amlenvrc')

import os
# Moving the original aml to aml_orig
aml_path = get_ipython().getoutput('which aml')
aml_path = aml_path[0]
aml_path_orig = aml_path + "_orig"
if not os.path.exists(aml_path_orig):
    get_ipython().system('mv $aml_path $aml_path_orig')

# Writing a new script to source the env variables
# before running the aml CLI
script = """
#!/bin/sh 
touch ~/.amlenvrc
. ~/.amlenvrc
export no_proxy=127.0.0.1
{} $@
""".format(aml_path_orig)
with open(aml_path, 'w') as f:
    f.write(script)

# Setting the permission to executable
get_ipython().system('chmod 755 $aml_path')

get_ipython().system('aml')

get_ipython().system('wget "https://migonzastorage.blob.core.windows.net/deep-learning/models/cntk/imagenet/ResNet_152.model"')

get_ipython().system('wget "https://ikcompuvision.blob.core.windows.net/acs/synset.txt"')

get_ipython().run_cell_magic('writefile', 'driver.py', 'import numpy as np\nimport logging\nimport sys\nimport json\nimport timeit as t\nimport urllib.request\nimport base64\nfrom cntk import load_model, combine\nfrom PIL import Image, ImageOps\nfrom io import BytesIO\n\nlogger = logging.getLogger("cntk_svc_logger")\nch = logging.StreamHandler(sys.stdout)\nlogger.addHandler(ch)\n\ntrainedModel = None\nmem_after_init = None\nlabelLookup = None\ntopResult = 3\n\ndef aml_cli_get_sample_request():\n    return \'{"input": ["base64Image"]}\'\n\ndef init():\n    global trainedModel, labelLookup, mem_after_init\n\n    # Load the model from disk and perform evals\n    # Load labels txt\n    with open(\'synset.txt\', \'r\') as f:\n        labelLookup = [l.rstrip() for l in f]\n    \n    # The pre-trained model was trained using brainscript\n    # Loading is not we need the right index \n    # See https://github.com/Microsoft/CNTK/wiki/How-do-I-Evaluate-models-in-Python\n    # Load model and load the model from brainscript (3rd index)\n    trainedModel = load_model(\'ResNet_152.model\')\n    trainedModel = combine([trainedModel.outputs[3].owner])\n\ndef run(inputString):\n\n    start = t.default_timer()\n\n    images = json.loads(inputString)\n    result = []\n    totalPreprocessTime = 0\n    totalEvalTime = 0\n    totalResultPrepTime = 0\n\n    for base64ImgString in images:\n\n        if base64ImgString.startswith(\'b\\\'\'):\n            base64ImgString = base64ImgString[2:-1]\n        base64Img = base64ImgString.encode(\'utf-8\')\n\n        # Preprocess the input data\n        startPreprocess = t.default_timer()\n        decoded_img = base64.b64decode(base64Img)\n        img_buffer = BytesIO(decoded_img)\n        # Load image with PIL (RGB)\n        pil_img = Image.open(img_buffer).convert(\'RGB\')\n        pil_img = ImageOps.fit(pil_img, (224, 224), Image.ANTIALIAS)\n        rgb_image = np.array(pil_img, dtype=np.float32)\n        # Resnet trained with BGR\n        bgr_image = rgb_image[..., [2, 1, 0]]\n        imageData = np.ascontiguousarray(np.rollaxis(bgr_image, 2))\n\n        endPreprocess = t.default_timer()\n        totalPreprocessTime += endPreprocess - startPreprocess\n\n        # Evaluate the model using the input data\n        startEval = t.default_timer()\n        imgPredictions = np.squeeze(trainedModel.eval(\n            {trainedModel.arguments[0]: [imageData]}))\n        endEval = t.default_timer()\n        totalEvalTime += endEval - startEval\n\n        # Only return top 3 predictions\n        startResultPrep = t.default_timer()\n        resultIndices = (-np.array(imgPredictions)).argsort()[:topResult]\n        imgTopPredictions = []\n        for i in range(topResult):\n            imgTopPredictions.append(\n                (labelLookup[resultIndices[i]], imgPredictions[resultIndices[i]] * 100))\n        endResultPrep = t.default_timer()\n        result.append(imgTopPredictions)\n\n        totalResultPrepTime += endResultPrep - startResultPrep\n\n    end = t.default_timer()\n\n    logger.info("Predictions: {0}".format(result))\n    logger.info("Predictions took {0} ms".format(\n        round((end - start) * 1000, 2)))\n    logger.info("Time distribution: preprocess={0} ms, eval={1} ms, resultPrep = {2} ms".format(round(\n        totalPreprocessTime * 1000, 2), round(totalEvalTime * 1000, 2), round(totalResultPrepTime * 1000, 2)))\n\n    actualWorkTime = round(\n        (totalPreprocessTime + totalEvalTime + totalResultPrepTime) * 1000, 2)\n    return (result, \'Computed in {0} ms\'.format(actualWorkTime))')

get_ipython().system('aml env setup -s $ACS_deployment_key')

get_ipython().system('cat ~/.amlenvrc')

get_ipython().system(' . ~/.amlenvrc && ssh-keyscan -p 2200 $AML_ACS_MASTER >> ~/.ssh/known_hosts')

get_ipython().system('cat ~/.amlenvrc')

get_ipython().system("echo '\\n' | aml env cluster")

service_name = 'cntkservice'
get_ipython().system('aml service create realtime -r cntk-py -f driver.py -m ResNet_152.model -d synset.txt -n $service_name')

get_ipython().system('aml service view realtime $service_name -v')

CLUSTER_SCORING_URL = "http://YOUR_SCORING_URL:9091/score"

if "YOUR_SCORING_URL" in CLUSTER_SCORING_URL:
    print("/!\ STOP /!\ You need to modify the value above to contain your scoring url")
else:
    print("You are good to go! :)")

import base64
import urllib
import requests
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from io import BytesIO
get_ipython().magic('matplotlib inline')

def url_img_to_json_img(url):
    bytfile = BytesIO(urllib.request.urlopen(url).read())
    img = Image.open(bytfile).convert('RGB')  # 3 Channels
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)  # Fixed size 
    plt.imshow(img)
    imgio = BytesIO()
    img.save(imgio, 'PNG')
    imgio.seek(0)
    dataimg = base64.b64encode(imgio.read())
    return json.dumps(
        {'input':'[\"{0}\"]'.format(dataimg.decode('utf-8'))})

HEADERS = {'content-type': 'application/json',
           'X-Marathon-App-Id': '/{}'.format(service_name)}

image_url = 'http://thomasdelteillondon.blob.core.windows.net/public/shuttle.jpg'
jsondata = url_img_to_json_img(image_url)

res = requests.post(CLUSTER_SCORING_URL, data=jsondata, headers=HEADERS)

print(json.dumps(res.json(), indent=4))

import random
import asyncio
from aiohttp import ClientSession
import json

async def fetch(url, session):
    async with session.post(url, headers={
        "content-type":"application/json",
        "X-Marathon-App-Id":"/{}".format(service_name)
    }, data=jsondata) as response:
        date = response.headers.get("DATE")
        #print("{}:{}".format(date, response.url))
        return await response.read()


async def bound_fetch(sem, url, session):
    # Getter function with semaphore.
    async with sem:
        await fetch(url, session)


async def run(r):
    url = CLUSTER_SCORING_URL
    tasks = []
    # create instance of Semaphore
    sem = asyncio.Semaphore(1000)

    # Create client session that will ensure we dont open new connection
    # per each request.
    async with ClientSession() as session:
        for i in range(r):
            # pass Semaphore and session to every GET request
            task = asyncio.ensure_future(bound_fetch(sem, url, session))
            tasks.append(task)

        responses = asyncio.gather(*tasks)
        await responses

get_ipython().run_cell_magic('time', '', 'number = 30\nloop = asyncio.get_event_loop()\n\nfuture = asyncio.ensure_future(run(number))\nloop.run_until_complete(future)')

resource_group = name+"rg"
get_ipython().system('az group delete --yes -n $resource_group')

get_ipython().system('ps aux | grep ssh')

get_ipython().system('cat ~/.ssh/acs_id_rsa')

