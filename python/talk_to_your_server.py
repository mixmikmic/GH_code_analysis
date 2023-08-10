from fabric.api import *

env.host_string="ec2-35-166-108-198.us-west-2.compute.amazonaws.com"
env.user = "ubuntu"
env.key_filename = ['/home/cognizac/Downloads/Stevens-key2.pem']

# run a command on the server to make sure you're connected
run('uname -a')

# stop the service for your project so you can edit the code
sudo('service myproject stop')

# stop nginx so you can use port 41593 for testing
sudo('service nginx stop')

# download the current code for your project
get('myproject.py')

# see what directory your currently in on the local machine
pwd

# cd into the directory created when you downloaded the project.py file
import os
os.chdir('ec2-35-166-108-198.us-west-2.compute.amazonaws.com/')

# checking to make sure we're in the right working directory
pwd

# list the files to see what's there
ls

# edit the code in a separate window, feel free to use my code
put('myproject.py')

# if you are using my code with TextBlob make sure to install TextBlob to the chatbot env
with prefix('source activate chatbot'):
    run('pip install textblob')

# we need to use our chatbot virtual environment to run the server code
with prefix('source activate chatbot'):
    run('python myproject.py')
    
#if you need to stop the server script, just select this cell and click the stop button at the top

# after you've stopped the server, make sure it's not still running remotely
sudo('killall python')

sudo('service myproject start')
sudo('service nginx start')



