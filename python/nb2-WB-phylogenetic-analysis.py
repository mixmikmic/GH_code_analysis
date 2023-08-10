## show the address of this git repo
get_ipython().system(' git config --get remote.origin.url')

## all necessary software is installed alongside ipyrad, 
## and can be installed by uncommenting the command below
# conda install -c ipyrad ipyrad -y

## import basic modules and ipyrad and print version
import os
import socket
import glob
import subprocess as sps
import numpy as np
import ipyparallel as ipp
import ipyrad as ip

print "ipyrad v.{}".format(ip.__version__)
print "ipyparallel v.{}".format(ipp.__version__)
print "numpy v.{}".format(np.__version__)

## open direct and load-balanced views to the client
ipyclient = ipp.Client()
lbview = ipyclient.load_balanced_view()
print "{} total cores".format(len(ipyclient.ids))

## confirm we are connected to 5 8-core nodes
hosts = ipyclient[:].apply_sync(socket.gethostname)

## get an engine id from each host to send threaded jobs to
threaded = {host:[] for host in set(hosts)}
for hid, host in enumerate(hosts):
    threaded[host].append(hid)
    
## print threaded setup, and save as threaded-views
tview = {}
idx = 0
for host, ids in threaded.items():
    print host, ids
    ## threaded-views
    tview[idx] = ipyclient.load_balanced_view(targets=ids)
    idx += 1

## create a new directory in HPC scratch dir
WORK = "/ysm-gpfs/scratch60/de243/WB-PED"
if not os.path.exists(WORK):
    os.mkdir(WORK)

## the current dir (./) in which this notebook resides
NBDIR = os.path.realpath(os.curdir)

## print both
print "working directory (WORK) = {}".format(WORK)
print "current directory (NBDIR) = {}".format(NBDIR)

NAMES = {"d33291": "P. oxycarpa 33291", 
         "d41389": "P. cranolopha 41389", 
         "d41237": "P. cranolopha 41237", 
         "d40328": "P. bidentata 40328",
         "d39531": "P. cranolpha 39531",
         "d31733": "P. latituba 31733",
         "d33291": "P. oxycarpa 33291", 
         "d39187": "P. souliei 39187", 
         "d39103": "P. decorissima 39103", 
         "d39253": "P. decorissima 39253",
         "decor21": "P. decorissima XX-DE21", 
         "d34041": "P. decorissima 34041",
         "d39114": "P. armata var. trimaculata 39114", 
         "d39404": "P. armata var. trimaculata 39404", 
         "d39968": "P. davidii 39968", 
         "d35422": "P. longiflora 35422", 
         "d41058": "P. longiflora var. tubiformis 41058", 
         "d39104": "P. longiflora var. tubiformis 39104", 
         "d19long1": "P. longiflora XX-DE19", 
         "d30695": "P. siphonantha 30695", 
         "d41732": "P. siphonantha 41732", 
         "d35178": "P. siphonantha 35178", 
         "d35371": "P. siphonantha 35371", 
         "d35320": "P. cephalantha 35320", 
         "d30181": "P. fletcheri 30181"
        }

## make raxml dir
RAXDIR = os.path.join(os.curdir, "analysis_raxml")
RAXDIR = os.path.realpath(RAXDIR)
if not os.path.exists(RAXDIR):
    os.mkdir(RAXDIR)
    
## get outgroup string from assembly object, or wherever
min4 = ip.load_json(os.path.join(WORK, "c85d5f2h5/min4_c85d5f2h5.json"))
OUT = ",".join([i for i in min4.samples.keys() if i[0] == "d"])

## run raxml in the background
cmd4 = ["/home2/de243/miniconda2/bin/raxmlHPC-PTHREADS", 
        "-f", "a", 
        "-m", "GTRGAMMA", 
        "-N", "100", 
        "-T", "16", 
        "-x", "12345", 
        "-p", "54321",
        "-o", OUT, 
        "-w", RAXDIR, 
        "-n", "min4_tree",
        "-s", os.path.join(NBDIR, "min4_c85d5f2h5_outfiles/min4_c85d5f2h5.phy")]
        
cmd10 = ["/home2/de243/miniconda2/bin/raxmlHPC-PTHREADS", 
        "-f", "a", 
        "-m", "GTRGAMMA", 
        "-N", "100", 
        "-T", "16", 
        "-x", "12345", 
        "-p", "54321",
        "-o", OUT, 
        "-w", RAXDIR, 
        "-n", "min10_tree",
        "-s", os.path.join(NBDIR, "min10_c85d5f2h5_outfiles/min10_c85d5f2h5.phy")]
        
## Send jobs to different hosts
asyncs = {}
asyncs["min4"] = tview[0].apply(sps.check_output, cmd4)
asyncs["min10"] = tview[1].apply(sps.check_output, cmd10)

## Check whether jobs have finished
for job, async in asyncs.items():
    if async.ready():
        if async.successful():
            print "job: [{}] finished.".format(job)
            print async.result()
        else:
            print async.exception()
    else:
        print "job: [{}]\t Elapsed: {:.0f}s".format(job, async.elapsed)



