## show that this dir is a git repo (has .git file mapping to the address shown)
## this allows me to push updates to this notebook directly to github, 
## to easily share its conents with others.
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

## confirm we are connected to 4 8-core nodes
hosts = ipyclient[:].apply_sync(socket.gethostname)
for hostname in set(hosts):
    print("host compute node: [{} cores] on {}"          .format(hosts.count(hostname), hostname))

## create a new directory in HPC scratch dir
WORK = "/ysm-gpfs/scratch60/de243/WB-PED"
if not os.path.exists(WORK):
    os.mkdir(WORK)

## the current dir (./) in which this notebook resides
NBDIR = os.path.realpath(os.curdir)

## print both
print "working directory (WORK) = {}".format(WORK)
print "current directory (NBDIR) = {}".format(NBDIR)

## Locations of the raw data stored temporarily on Yale's Louise HPC cluster
## Data are also stored more permanently on local computer tinus at Yale
RAWREADS = "/ysm-gpfs/project/de243/RADSEQ_RAWS/WB-PEDICULARIS/*.fastq.gz"
BARCODES = "/ysm-gpfs/project/de243/BARCODES/WB-PED_barcodes.txt"

## if needed, uncomment and run the command below to install fastqc
#conda install -c bioconda fastqc -q 

## create a tmp directory for fastqc outfiles (./tmp_fastqc)
QUALDIR = os.path.join(NBDIR, "fastqc")
if not os.path.exists(QUALDIR):
    os.mkdir(QUALDIR)
    
## run fastqc on all raw data files and write outputs to fastqc.
## This is parallelized by load-balancing by lbview 
jobs = {}
for infile in glob.glob(RAWREADS):
    cmd = ['fastqc', infile, '--outdir', QUALDIR, '-t', '1', '-q']
    jobs[infile] = lbview.apply_async(sps.check_output, cmd)
    
## block until all finished and print progress
ipyclient.wait_interactive()

## check for fails
for key in jobs:
    if not jobs[key].successful():
        print jobs[key].exception()

## create an object to demultiplex each lane
demux = ip.Assembly("WB-PED_demux")

## set basic derep parameters for the two objects
demux.set_params("project_dir", os.path.join(WORK, "demux_reads"))
demux.set_params("raw_fastq_path", RAWREADS)
demux.set_params("barcodes_path", BARCODES)
demux.set_params("max_barcode_mismatch", 0)
demux.set_params("datatype", "pairgbs")
demux.set_params("restriction_overhang", ("TGCAG", "TGCAG"))

## print params
demux.get_params()

demux.run("1")

## print total
print "total reads recovered: {}\n".format(demux.stats.reads_raw.sum())

## print header, and then selected results across raw files
get_ipython().system(' head -n 1 $demux.stats_files.s1')
get_ipython().system(' cat $demux.stats_files.s1 | grep 0[4-5][0-9].fastq')

## run with one bp mismatch
demux.set_params("max_barcode_mismatch", 1)
## the force flag tells it to overwrite the previous data
demux.run("1", force=True)

## print total
print "total reads recovered: {}\n".format(demux.stats.reads_raw.sum())

## print header, and then selected results across raw files
get_ipython().system(' head -n 1 $demux.stats_files.s1')
get_ipython().system(' cat $demux.stats_files.s1 | grep 0[4-5][0-9].fastq')

## the result of this demux look better, so I copied the step1
## stats file to the local dir and pushed it to the git repo.
get_ipython().system(' cp $demux.stats_files.s1 $NBDIR')

## this will be our assembly object for steps 1-6
data = ip.Assembly("c85d5f2h5")

## (optional) set a more fine-tuned threading for our cluster
data._ipcluster["threads"] = 4

## demux data location
DEMUX = os.path.join(demux.dirs.fastqs, "*gz")

## set parameters for this assembly and print them 
data.set_params("project_dir", os.path.join(WORK, data.name))
data.set_params("sorted_fastq_path", DEMUX)
data.set_params("barcodes_path", BARCODES)
data.set_params("filter_adapters", 2)
data.set_params("datatype", "pairgbs")
data.set_params("restriction_overhang", ("TGCAG", "TGCAG"))
data.set_params("max_Hs_consens", (5, 5))
data.set_params("max_SNPs_locus", (10, 10))
data.set_params("trim_overhang", (5, 5, 5, 5))
data.get_params()

## run steps 1-6
data.run("12", force=True)

print data.stats.describe().astype(int)

## print just the first few samples
print data.stats_dfs.s2.head()

data.run("3456")

## quick look at which filters applied most in step 5
print data.stats_dfs.s5.describe().astype(int)

## create named branches for final assemblies
min4 = data.branch("min4_c85d5f2h5")
min4.set_params("min_samples_locus", 4)

min10 = data.branch("min10_c85d5f2h5")
min10.set_params("min_samples_locus", 10)

## assemble outfiles
min4.run("7", force=True)
min10.run("7", force=True)

get_ipython().system('cat $min4.stats_files.s7')

get_ipython().system(' head -n 50 $min4.outfiles.loci | cut -c 1-80')

## reload, b/c I disconnected and came back.
min10 = ip.load_json("/fastscratch/de243/WB-PED/c85d5f2h5/min10_c85d5f2h5.json")
min10.outfiles.loci[:-5]+'.phy'



## ask if it's still running in background
if proc.poll():
    print proc.returncode
else:
    tail = get_ipython().getoutput('tail $RAXDIR/RAxML_info.min10*')
    print "still running. \ntail:", tail[-1]

## move data from WORK to NBDIR
print WORK
print NBDIR

get_ipython().system(' cp -r  $WORK/c85d5f2h5/min4_c85d5f2h5_outfiles/ $NBDIR')
get_ipython().system(' cp -r  $WORK/c85d5f2h5/min10_c85d5f2h5_outfiles/ $NBDIR')

import ete3 as ete

## load tree
tre = ete.Tree("analysis_raxml/RAxML_bipartitions.min4_tree", format=0)

## sub in names
for node in tre.traverse():
    if node.name in NAMES:
        node.name = NAMES[node.name]


#print tre.get_ascii(attributes=["name", "color"], show_internal=False)
tre.ladderize()
print tre.get_ascii(attributes=["name", "support"])

test = ip.load_json("/fastscratch/de243/WB-PED/c85d5f2h5/c85d5f2h5.json")
sample = test.samples["d19long1"]

