# Make sure we're in SageMaker root
get_ipython().run_line_magic('cd', '~/SageMaker')

# Remove any old versions of the library
get_ipython().system('rm -rf ~/SageMaker/donkey')

# Clone the Donkey library git
get_ipython().system('git clone https://github.com/wroscoe/donkey.git')

# Update Donkey dependencies

# Keras is pinned to version 2.0.8 in the Donkey requirements. Change this to allow a newer version
get_ipython().system("sed -i -e 's/keras==2.0.8/keras>=2.1.2/g' ~/SageMaker/donkey/setup.py")
get_ipython().system("sed -i -e 's/tensorflow>=1.1/tensorflow-gpu>=1.4/g' ~/SageMaker/donkey/setup.py")

# Install
get_ipython().system('pip uninstall --yes donkeycar')
get_ipython().system('pip install ~/SageMaker/donkey')

# Create a new car using the library CLI
get_ipython().system('rm -rf ~/d2')
get_ipython().system('donkey createcar --path ~/d2')

from sagemaker import get_execution_role

# Bucket location to get training data
sample_data_location = 's3://jayway-robocar-raw-data/samples'

# Create a data directory
get_ipython().system('mkdir -pv ~/SageMaker/data')
get_ipython().system('aws s3 cp {sample_data_location}/ore.zip ~/SageMaker/data')

# Unzip to data dir
get_ipython().system('unzip -o ~/SageMaker/data/ore.zip -d ~/SageMaker/data')

# check tub size
get_ipython().system('du -h ~/SageMaker/data')

get_ipython().run_cell_magic('time', '', "\n# Make sure we're in Donkey car root\n%cd ~/d2\n\n# Start the training\n!python manage.py train --tub='../SageMaker/data/tub_8_18-02-09' --model './models/my-first-model'")

get_ipython().system('ls -la ./models')

import sagemaker
dl_bucket = sagemaker.Session().default_bucket()

# push the model to your bucket
get_ipython().system('aws s3 cp ./models/my-first-model s3://{dl_bucket}/models/my-first-model')

