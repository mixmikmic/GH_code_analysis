"""
Get tuples from the classify file where the members are
(genbank_id, gene_start, gene_end, context_gene_type)
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os
os.chdir("/home/nafizh/Context_gene_mining/new_pipeline/")

in_handle = open("classify", "r")
gene_list = [] # the members will be 4 item tuples

for index, record in enumerate(SeqIO.parse(in_handle, "fasta")):
    context_gene_type = record.description.split("|")[-1]
    context_gene_type = context_gene_type.rstrip()
    genbank_id = record.description.split("|")[4]
    gene_start = record.description.split("|")[-6]
    gene_end = record.description.split("|")[-5]
    
    gene_list.append((genbank_id, gene_start, gene_end, context_gene_type))
#     print genbank_id, context_gene_type, gene_start, gene_end
    
#     if index == 4:
#         break

gene_list_set = set(gene_list)

print len(gene_list)
print len(gene_list_set)
print "Done"
    

"""
From the list of tuples that I get at the previous step, find those genes in their respective
genbank file and try to find the annotations attached to them.
"""

from collections import defaultdict

keyword_dic = defaultdict(list)


def find(name, path = "/home/nafizh/bacteria_archaea_genbank_files"):
    """finds the path of the genbank file"""
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

# Get the annotation for each context gene and store it inside keyword_dic
for index, tup in enumerate(gene_list_set):
    print tup
    tok = tup[0].split(".")[0] + ".gbk"
    genbank_file = find(tok)
    #print genbank_file
    
    record = SeqIO.parse(open(genbank_file), "genbank").next()
    
    for feature in record.features:
        if feature.type == 'CDS':
            if int(tup[1]) >= feature.location.start.position and int(tup[2]) <= feature.location.end.position:
                #print feature.qualifiers.get('note', ['no note'])[0]
                #print feature.qualifiers.get('product', ['no product'])[0]
                keyword_dic[tup[3]].append(feature.qualifiers.get('product', ['no product'])[0])
    
print keyword_dic['modifier']
print keyword_dic['transport']

import subprocess

cmd = ['locate', 'NC_006322.gbk']
proc = subprocess.Popen(cmd, stdout = subprocess.PIPE)
var = proc.stdout.read()
print var

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        
print find('NC_006322.gbk', '/home/nafizh/bacteria_archaea_genbank_files/')

from collections import Counter
print Counter(keyword_dic['modifier'])

from collections import Counter
print Counter(keyword_dic['transport'])

from collections import Counter
print Counter(keyword_dic['immunity'])

from collections import Counter
print Counter(keyword_dic['regulator'])



