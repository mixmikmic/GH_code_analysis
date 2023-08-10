get_ipython().system('apt-get update')

get_ipython().system('apt-get install ant mercurial openjdk-8-jdk -y')

get_ipython().system('hg clone http://hg-iesl.cs.umass.edu/hg/mallet')

import os
os.chdir('/sharedfolder/mallet')
get_ipython().system('ant')

import os
os.chdir('/sharedfolder/mallet')
get_ipython().system('ls bin')

