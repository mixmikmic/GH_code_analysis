#  Source: https://raw.githubusercontent.com/joshisa/jspark/master/jspark/ipfs.py

# Useful Reading
#   -   https://ipfs.io/ipfs/QmNhFJjGcMPqpuYfxL62VVB9528NXqDNMFXiqN5bgFYiZ1/its-time-for-the-permanent-web.html
#   -   https://netninja.com/2015/09/30/ipfs_interplanetary_file_system/

# Purpose:  Installation of IPFS (InterPlanetary File System) within Jupyter
#           IPFS is a peer-to-peer distributed file system that seeks to 
#           connect all computing devices with the same system of files.
#           In some ways, IPFS is similar to the World Wide Web, but IPFS 
#           could be seen as a single BitTorrent swarm, exchanging objects 
#           within one Git repository. IPFS has no single point of failure,
#           and nodes do not need to trust each other.
#           The filesystem can be accessed in a variety of ways, including 
#           via FUSE and over HTTP. A local file can be added to the IPFS 
#           filesystem, making it available to the world. Files are identified 
#           by their hashes, so it's caching-friendly. They are distributed 
#           using a BitTorrent-based protocol. (source: Wikipedia)
# Status:  Alpha (Experimental)
# Use Case Possibilities:  
#   1. Easier sharing of data analysis result sets
#   2. Easier access to popular open data sets via a permanent url
#   3. Easier transfer of assets between notebook servers on different infra
# Invoke Command: 
#   %load https://raw.githubusercontent.com/joshisa/jspark/master/jspark/ipfs.py
#
# Author:  Sanjay Joshi  joshisa (at) us(dot)ibm(dot)com
# Author:  Chris Waldon ckwaldon (at) us(dot)ibm(dot)com
# License: Apache 2.0
# Organization:  IBM jStart (http://ibm.com/jstart)

import sys
import path
import os
import subprocess as sub
import signal
import time
from IPython.display import *

# Print Working Directory
prefix = os.getcwd()
proposed = os.path.dirname(os.path.dirname(prefix))
if os.access(proposed, os.W_OK):
    print("Prefix proposal accepted")
    prefix = proposed
elif os.access(prefix, os.W_OK):
    print("Prefix original accepted")
    prefix = prefix
else:
    sys.exit("No writeable directory found")

# Let's setup useful paths
localDir = prefix + "/.local"
shareDir = prefix + "/.local/share"
ipfsDir = shareDir + "/ipfs"
ipfsHomeDir = ipfsDir + "/go-ipfs"
ipfsRepoDir = shareDir + "/ipfsrepo"

# Let's make sure all of the paths are created if they don't exist
get_ipython().system('mkdir $localDir 2> /dev/null')
get_ipython().system('mkdir $shareDir 2> /dev/null')
get_ipython().system('mkdir $ipfsDir 2> /dev/null')
get_ipython().system('mkdir $ipfsRepoDir 2> /dev/null')
get_ipython().system('rm $ipfsRepoDir/repo.lock')
get_ipython().system('kill $(pgrep ipfs)')

# Let's now define some IMPORTANT env vars
os.environ["PATH"] += os.pathsep + ipfsHomeDir
os.environ["IPFS_PATH"] = ipfsRepoDir
os.environ["IPFS_LOGGING"] = "" #<empty>, info, error, debug

print("prefix = " + prefix)
print("shareDir = " + shareDir)
print("ipfs Dir = " + ipfsDir)
print("IPFS Repo Dir = " + ipfsRepoDir) 

# Define an easy way to run terminal commands
def run_cmd(cmd):
    p = sub.Popen(cmd, stdout=sub.PIPE,
               stderr=sub.PIPE)
    out, err = p.communicate()
    try:
        out = out.decode("utf-8")
        err = err.decode("utf-8")
    except:
        pass
    print(err)
    print(out)

# Define an IPFS Helper Class
class ipfs():
    def __init__(self):
        run_cmd(['ipfs', 'version'])
        run_cmd(['ipfs', 'init'])
        self.daemonStart()
        
    def setLog(self, loglevel=""):
        os.environ["IPFS_LOGGING"] = loglevel
        
    def daemonStart(self):
        p = sub.Popen("nohup ipfs daemon > nohup.out 2>&1 &", shell=True)
        (result, err) = p.communicate()
        time.sleep(5)
        output = get_ipython().getoutput('cat nohup.out')
        log = '\n'.join(str(x) for x in output)
        print(log)

    def daemonStop(self):
        PID = get_ipython().getoutput('ps -ef | grep "\\bipfs daemon\\b" | awk \'{print $2}\'')
        os.kill(int(PID[0]), signal.SIGTERM)
        time.sleep(5)
        log = get_ipython().getoutput('cat nohup.out')
        log = '\n'.join(str(x) for x in output)
        print(log)
    
# Let's test to see if ipfs already exists in this notebook workspace
isIPFSInstalled = os.path.isfile(ipfsHomeDir + "/ipfs")
if isIPFSInstalled:
    print("Congratulations! IPFS is already installed within your notebook user space")
else:
    print("IPFS is NOT installed within this notebook's user space")
    print("Initiating installation sequence ...")
    
    print("    Downloading and Installing the IPFS binary")
    get_ipython().system('wget https://dist.ipfs.io/go-ipfs/v0.4.3/go-ipfs_v0.4.3_linux-amd64.tar.gz -O $ipfsDir/go-ipfs_v0.4.3_linux-amd64.tar.gz')
    get_ipython().system('tar -zxvf $ipfsDir/go-ipfs_v0.4.3_linux-amd64.tar.gz -C $ipfsDir >/dev/null')
    # Retest ipfs existence
    isIPFSInstalled = os.path.isfile(ipfsHomeDir + "/ipfs")
    print("Congratulations!! IPFS is now installed within your notebook")
    print("To validate, run the following command within a cell:")
    print("")
    print("        ipfs = ipfs()")

daemon = ipfs()

# If you have an IPFS version other than v0.4.3, please delete it using the following and rerun the first cell
# !rm -rf $ipfsDir/*

get_ipython().system('pip install ipfsapi --user')
import ipfsapi
api = ipfsapi.Client("localhost", 5001)

print api.cat("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG/readme")

api.swarm_peers()

api.id()

api.get("QmSeYATNaa2fSR3eMqRD8uXwujVLT2JU9wQvSjCd1Rf8pZ/1566 - Board Game/1566 - Board Game.png")
Image("1566 - Board Game.png")

# Substitute your comic's number and name!
api.get("QmSeYATNaa2fSR3eMqRD8uXwujVLT2JU9wQvSjCd1Rf8pZ/comicnumber - comic name/comicnumber - comic name.png")
Image("comicnumber - comic name.png")

api.get("QmVfFbASG11MjUFuvfAJmjKpxRGYstCAw9GmZEpbA7KE7A")
get_ipython().system('mv QmVfFbASG11MjUFuvfAJmjKpxRGYstCAw9GmZEpbA7KE7A QmVfFbASG11MjUFuvfAJmjKpxRGYstCAw9GmZEpbA7KE7A.jpg')
Image("QmVfFbASG11MjUFuvfAJmjKpxRGYstCAw9GmZEpbA7KE7A.jpg")

api.add("QmVfFbASG11MjUFuvfAJmjKpxRGYstCAw9GmZEpbA7KE7A.jpg")



