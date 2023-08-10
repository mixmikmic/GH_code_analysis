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

## open a view to the client
ipyclient = ipp.Client()

## confirm we are connected to 4 8-core nodes
hosts = ipyclient[:].apply_sync(socket.gethostname)
for hostname in set(hosts):
    print("host compute node: [{} cores] on {}"          .format(hosts.count(hostname), hostname))

## create a new working directory in HPC scratch dir
WORK = "/fastscratch/de243/WB-PED"
if not os.path.exists(WORK):
    os.mkdir(WORK)

## the current dir (./) in which this notebook resides
NBDIR = os.path.realpath(os.curdir)

## print it
print "working directory (WORK) = {}".format(WORK)
print "current directory (NBDIR) = {}".format(NBDIR)

## Locations of the raw data stored temporarily on Yale's Louise HPC cluster
## Data are also stored more permanently on local computer tinus at Yale
RAWREADS = "/fastscratch/de243/TMP_RAWS/*.fastq.gz"
BARCODES = "/fastscratch/de243/TMP_RAWS/WB-PED_barcodes.txt"

## uncomment this to install fastqc with conda
#conda install -c bioconda fastqc -q 

## create a tmp directory for fastqc outfiles (./tmp_fastqc)
QUALDIR = os.path.join(NBDIR, "tmp_fastqc")
if not os.path.exists(QUALDIR):
    os.mkdir(QUALDIR)
    
## run fastqc on all raw data files and write outputs to fastqc tmpdir.
## This is parallelized by load-balancing with ipyclient
lbview = ipyclient.load_balanced_view()
for rawfile in glob.glob(RAWREADS):
    cmd = ['fastqc', rawfile, '--outdir', QUALDIR, '-t', '1', '-q']
    lbview.apply_async(sps.check_output, cmd)
    
## block until all finished and print progress
ipyclient.wait_interactive()

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
## demux.get_params()

demux.run("1")

## print total
print "total reads recovered: {}\n".format(demux.stats.reads_raw.sum())

## print header, and then selected results across raw files
get_ipython().system(' head -n 1 $demux.stats_files.s1')
get_ipython().system(' cat $demux.stats_files.s1 | grep 0[4-5][0-9].fastq')

## run with one bp mismatch
demux.set_params("max_barcode_mismatch", 1)
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
data.set_params("trim_overhang", (0, 5, 5, 0))
data.get_params()

## run steps 1-6
data.run("12")

## print just the first few samples
print data.stats_dfs.s2.head()

data.run("3456")

## create named branches for final assemblies
min4 = data.branch("min4_c85d5f2h5")
min4.set_params("min_samples_locus", 4)

min10 = data.branch("min10_c85d5f2h5")
min10.set_params("min_samples_locus", 10)

## assemble outfiles
min4.run("7", force=True)
min10.run("7", force=True)

get_ipython().system('cat $min4.stats_files.s7')

get_ipython().run_cell_magic('bash', '', 'head -n 100 $min4.outfiles.loci | cut -c 1-80')

## reload, b/c I disconnected and came back.
min4 = ip.load_json("/fastscratch/de243/WB-PED/c85d5f2h5/min4_c85d5f2h5.json")
min4.outfiles.loci[:-5]+'.phy'

## make raxml dir
raxdir = os.path.join(os.curdir, "analysis_raxml")
raxdir = os.path.realpath(raxdir)
if not os.path.exists(raxdir):
    os.mkdir(raxdir)
    
## get outgroup string
OUT = ",".join([i for i in min4.samples.keys() if i[0] == "d"])

## run raxml in the background
cmd = ["/home2/de243/miniconda2/bin/raxmlHPC-PTHREADS", 
        "-f", "a", 
        "-m", "GTRGAMMA", 
        "-N", "100", 
        "-T", "8", 
        "-x", "12345", 
        "-p", "54321",
        "-o", OUT, 
        "-w", raxdir, 
        "-n", "min4_tree",
        "-s", min4.outfiles.loci[:-5]+'.phy']

## start process running in background
proc = sps.Popen(cmd, stderr=sps.PIPE, stdout=sps.PIPE)

## ask if it's still running in background
if proc.poll():
    print sps.returncode
else:
    tail = get_ipython().getoutput('tail $raxdir/*info*')
    print "still running: \n", tail[-1]

get_ipython().magic('load_ext rpy2.ipython')



