# import biopython
from Bio import SeqIO


def geneCount (filename):
    return(len(list(SeqIO.parse(filename, "fasta"))))

# apply gene Count to all files in each folder
methodList = ["genemark", "glimmer", "prodigal"] #list of methods (identical to folder name)
returnList = []
from os import listdir
from os.path import isfile, join
for a in methodList:
    onlyfiles = [f for f in listdir(a) if isfile(join(a, f))]
    count = {}
    for b in onlyfiles:
        count[b] = geneCount(a + '\\' + b)
    returnList.append(count)



returnList





