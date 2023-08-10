base_dir='/tmp/fashion'
dataset_dir='https://workshopml.spock.cloud/datasets/fashion-mnist'
pre_trained_model='https://workshopml.spock.cloud/models/fashion-mnist/model.tar.gz'

get_ipython().system('mkdir -p $base_dir/samples')
get_ipython().system('curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz | gunzip > $base_dir/samples/train-images-idx3-ubyte')
get_ipython().system('curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz | gunzip > $base_dir/samples/train-labels-idx1-ubyte')
get_ipython().system('curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz | gunzip > $base_dir/samples/t10k-images-idx3-ubyte')
get_ipython().system('curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz | gunzip > $base_dir/samples/t10k-labels-idx1-ubyte')
get_ipython().system('ls -lat $base_dir/samples/')

get_ipython().system('mkdir -p $base_dir/fashion_mnist')

import os
categories = ['TShirtTop', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','AnkleBoot' ]

for i in categories:
    try:
        os.mkdir(base_dir + '/fashion_mnist/%s' % i)
    except OSError as e:
        print(e)

get_ipython().system('pip install python-mnist')
from mnist import MNIST
from PIL import Image
import numpy as np

mndata = MNIST(base_dir + '/samples')
counter = 0
images, labels = mndata.load_training()
for i, img in enumerate(images):
    img = np.reshape(img, (28, 28))
    img = Image.fromarray(np.uint8(np.array(img)))
    img = img.convert("RGB")
    img.save(base_dir + '/fashion_mnist/%s/img_%d.jpg' % (categories[labels[i]], counter ))
    counter += 1

images, labels = mndata.load_testing()
for i, img in enumerate(images):
    img = np.reshape(img, (28, 28))
    img = Image.fromarray(np.uint8(np.array(img)))
    img = img.convert("RGB")
    img.save(base_dir + '/fashion_mnist/%s/img_%d.jpg' % (categories[labels[i]], counter ))
    counter += 1

get_ipython().system('ls -lat $base_dir/fashion_mnist/')

# Here we will search for the python script im2rec
import sys,os

suffix='/mxnet/tools/im2rec.py'
im2rec = list(filter( (lambda x: os.path.isfile(x + suffix )), sys.path))[0] + suffix
get_ipython().run_line_magic('env', 'IM2REC=$im2rec')
get_ipython().run_line_magic('env', 'BASE_DIR=$base_dir')

get_ipython().run_cell_magic('bash', '', '\ncd $BASE_DIR\npython $IM2REC --list=1 --recursive=1 --shuffle=1 --test-ratio=0.3 --train-ratio=0.7 fashion_mnist fashion_mnist/\nls *.lst')

get_ipython().run_cell_magic('bash', '', '\ncd $BASE_DIR\npython $IM2REC --num-thread=4 --pass-through=1 fashion_mnist_train.lst fashion_mnist\npython $IM2REC --num-thread=4 --pass-through=1 fashion_mnist_test.lst fashion_mnist\nls *.rec')

import sagemaker

# Get the current Sagemaker session
sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

train_path = sagemaker_session.upload_data(path=base_dir + '/fashion_mnist_train.rec', key_prefix='fashion_mnist/train')
test_path = sagemaker_session.upload_data(path=base_dir + '/fashion_mnist_test.rec', key_prefix='fashion_mnist/test')

get_ipython().run_cell_magic('time', '', "import boto3\nimport re\nimport os\nimport time\n\nfrom time import gmtime, strftime\nfrom sagemaker import get_execution_role\n\n# 1. Obtaining the role you already configured for Sagemaker when you setup\n# your Instance notebook (https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)\nrole = get_execution_role()\n\n# 2. The S3 Bucket that will store the dataset and the trained model\n# It was already defined above, while we uploaded the RecordIO files to the S3 bucket.\n\n# 3. Select the correct Docker image with the Image Classification algorithm\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]\nprint(training_image)")

# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 152 layers
num_layers = 152
# we need to specify the input image shape for the training data
image_shape = "3,28,28"
# we also need to specify the number of training samples in the training set
# for fashion_mnist it is 70012
num_training_samples = 70012
# specify the number of output classes
num_classes = 10
# batch size for training
mini_batch_size = 1024
# number of epochs
epochs = 1
# learning rate
learning_rate = 0.00001
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
# initialized with pre-trained weights
use_pretrained_model = 1
# Training algorithm/optimizer. Default is SGD
optimizer = 'sgd'

dataset_prefix='fashion_mnist'
# create unique job name 
job_name_prefix = 'fashion-mnist'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
job_name = job_name_prefix + timestamp

training_params = {}

# Here we set the reference for the Image Classification Docker image, stored on ECR (https://aws.amazon.com/pt/ecr/)
training_params["AlgorithmSpecification"] = {
    "TrainingImage": training_image,
    "TrainingInputMode": "File"
}

# The IAM role with all the permissions given to Sagemaker
training_params["RoleArn"] = role

# Here Sagemaker will store the final trained model
training_params["OutputDataConfig"] = {
    "S3OutputPath": 's3://{}/{}/output'.format(sagemaker_session.default_bucket(), job_name_prefix)
}

# This is the config of the instance that will execute the training
training_params["ResourceConfig"] = {
    "InstanceCount": 1,
    "InstanceType": "ml.p2.xlarge",
    "VolumeSizeInGB": 50
}

# The job name. You'll see this name in the Jobs section of the Sagemaker's console
training_params["TrainingJobName"] = job_name

# Here you will configure the hyperparameters used for training your model.
training_params["HyperParameters"] = {
    "image_shape": image_shape,
    "num_layers": str(num_layers),
    "num_training_samples": str(num_training_samples),
    "num_classes": str(num_classes),
    "mini_batch_size": str(mini_batch_size),
    "epochs": str(epochs),
    "learning_rate": str(learning_rate),
    "use_pretrained_model": str(use_pretrained_model),
    "optimizer": optimizer
}

# Training timeout
training_params["StoppingCondition"] = {
    "MaxRuntimeInSeconds": 360000
}

# The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)
training_params["InputDataConfig"] = []

# Please notice that we're using application/x-recordio for both 
# training and validation datasets, given our dataset is formated in RecordIO

# Here we set training dataset
# Training data should be inside a subdirectory called "train"
training_params["InputDataConfig"].append({
    "ChannelName": "train",
    "DataSource": {
        "S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": train_path,
            "S3DataDistributionType": "FullyReplicated"
        }
    },
    "ContentType": "application/x-recordio",
    "CompressionType": "None"
})

# Here we set validation dataset
# Validation data should be inside a subdirectory called "validation"
training_params["InputDataConfig"].append({
    "ChannelName": "validation",
    "DataSource": {
        "S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": test_path,
            "S3DataDistributionType": "FullyReplicated"
        }
    },
    "ContentType": "application/x-recordio",
    "CompressionType": "None"
})

print('Training job name: {}'.format(job_name))
print('\nInput Data Location: {}'.format(training_params['InputDataConfig'][0]['DataSource']['S3DataSource']))

import botocore
# Get the Sagemaker client
sagemaker = boto3.client(service_name='sagemaker')

# create the Amazon SageMaker training job

sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))

# If you skip the last session, keep this variable=True, False otherwise
use_pretrained_model=True

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nmodel_name="fashion-mnist"\nprint(model_name)\nif use_pretrained_model:\n    default_bucket=sagemaker_session.default_bucket()\n    prefix="fashion-mnist/model/model.tar.gz"\n    model_data="s3://{}/{}".format(default_bucket, prefix)\n    s3 = boto3.client(\'s3\')\n    resp = s3.list_objects(Bucket=default_bucket, Prefix=prefix)\n    if resp.get("Contents") is None:\n        print("Please wait. It will take around 6mins")\n        !curl -s $pre_trained_model | aws s3 cp - s3://$default_bucket/fashion-mnist/model/model.tar.gz\nelse:\n    info = sagemaker.describe_training_job(TrainingJobName=job_name)\n    model_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\n    print(model_data)\n\nprimary_container = {\n    \'Image\': training_image,\n    \'ModelDataUrl\': model_data,\n}\n\ntry:\n    sagemaker.create_model(\n        ModelName = model_name,\n        ExecutionRoleArn = role,\n        PrimaryContainer = primary_container)\nexcept botocore.exceptions.ClientError as e:\n    print(e.response[\'Error\'][\'Message\'])')

from time import gmtime, strftime

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
endpoint_config_name = job_name_prefix + '-epc-' + timestamp
endpoint_config_response = sagemaker.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.c4.2xlarge',
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print('Endpoint configuration name: {}'.format(endpoint_config_name))
print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))

get_ipython().run_cell_magic('time', '', "import time\n\ntimestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())\nendpoint_name = job_name_prefix + '-ep-' + timestamp\nprint('Endpoint name: {}'.format(endpoint_name))\n\nendpoint_params = {\n    'EndpointName': endpoint_name,\n    'EndpointConfigName': endpoint_config_name,\n}\nendpoint_response = sagemaker.create_endpoint(**endpoint_params)\nprint('EndpointArn = {}'.format(endpoint_response['EndpointArn']))\n\n\n# get the status of the endpoint\nresponse = sagemaker.describe_endpoint(EndpointName=endpoint_name)\nstatus = response['EndpointStatus']\nprint('EndpointStatus = {}'.format(status))\n\n\n# wait until the status has changed\nsagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)\n\n\n# print the status of the endpoint\nendpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)\nstatus = endpoint_response['EndpointStatus']\nprint('Endpoint creation ended with EndpointStatus = {}'.format(status))\n\nif status != 'InService':\n    raise Exception('Endpoint creation failed.')")

# Download test data
import mxnet as mx

get_ipython().system('mkdir -p $base_dir/test_data')
for i in range(5):
    mx.test_utils.download(dataset_dir + '/test_data/item%d_thumb.jpg' % (i+1), dirname=base_dir + '/test_data'),

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from PIL import Image

test_categories = ['Shirt','TShirtTop', 'AnkleBoot', 'Sneaker', 'Bag']

f, axarr = plt.subplots(1, 5, figsize=(20,12))
col = 0
for i in range(5):
    im = Image.open(base_dir + '/test_data/item%d_thumb.jpg' % (i+1))
    axarr[col].text(0, 0, '%s' %(test_categories[i] ), fontsize=15, color='blue')
    frame = axarr[col].imshow(im)
    col += 1
plt.show()

import json
import numpy as np
from io import BytesIO

runtime = boto3.Session().client(service_name='sagemaker-runtime') 
object_categories = ['AnkleBoot','Bag','Coat','Dress','Pullover','Sandal','Shirt','Sneaker','TShirtTop','Trouser']

_, axarr = plt.subplots(1, 5, figsize=(20,12))
col = 0
for i in range(5):
    
    # Load the image bytes
    img = open(base_dir + '/test_data/item%d_thumb.jpg' % (i+1), 'rb').read()
    
    # Call your model for predicting which object appears in this image.
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='application/x-image', 
        Body=bytearray(img)
    )
    # read the prediction result and parse the json
    result = response['Body'].read()
    result = json.loads(result)
    
    # which category has the highest confidence?
    pred_label_id = np.argmax(result)
    
    # Green when our model predicted correctly, otherwise, Red
    text_color = 'red'
    if object_categories[pred_label_id] == test_categories[i]:
        text_color = 'green'

    # Render the text for each image/prediction
    output_text = '%s (%f)' %(object_categories[pred_label_id], result[pred_label_id] )
    axarr[col].text(0, 0, output_text, fontsize=15, color=text_color)
    print( output_text )
    
    # Render the image
    img = Image.open(BytesIO(img))
    frame = axarr[col].imshow(img)
    
    col += 1
plt.show()

sagemaker.delete_endpoint(EndpointName=endpoint_name)

get_ipython().system('rm -rf $base_dir')

