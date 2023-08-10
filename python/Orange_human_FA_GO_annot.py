#setup

import urllib.request
import json 
import datetime
import pandas as pd
import numpy as np
import array
import matplotlib
import seaborn

from IPython.core.display import display, HTML

print(datetime.datetime.now().time())

GO_data = []
with urllib.request.urlopen("https://api.monarchinitiative.org/api/mart/gene/function/NCBITaxon:9606") as url:
    GO_data.append(json.loads(url.read().decode()))

print(datetime.datetime.now().time())

GO_data[0][0]

print(datetime.datetime.now().time())
FA_core = []
with urllib.request.urlopen("https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_1_core_complex.txt") as url:
    FA_core.append(url.read())
print(datetime.datetime.now().time())

print(datetime.datetime.now().time())
FA_effector = []
with urllib.request.urlopen("https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_2_effector_proteins.txt") as url:
    FA_effector.append(url.read())
print(datetime.datetime.now().time())

print(datetime.datetime.now().time())
FA_assoc = []
with urllib.request.urlopen("https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_3_associated_proteins.txt") as url:
    FA_assoc.append(url.read())
print(datetime.datetime.now().time())

print(datetime.datetime.now().time())
FA_all = []
with urllib.request.urlopen("https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_4_all_genes.txt") as url:
    FA_all.append(url.read())
print(datetime.datetime.now().time())

FA_core_list= FA_core[0].splitlines()
#print( len(FA_core_list))

gene_GO_dict = dict()

for i in range(0, len(FA_core_list)):
    curlist = FA_core_list[i].split()
    curid = curlist[0].decode("utf-8") 
    print(curid)
    #for j in range(0, len(GO_data[0])):
        #print(GO_data[0][j]['subject_label'] )
        
    qurl = "https://api.monarchinitiative.org/api/bioentity/gene/"+curid+"/function/"
    print(qurl)
    with urllib.request.urlopen(qurl) as thisurl:
        getdata = json.loads(thisurl.read().decode())

        for j in range(0, len(getdata['associations'])):
            #print(getdata['associations'][j]['object']['id'])
            if curlist[1] not in gene_GO_dict:
                gene_GO_dict[curlist[1]] = getdata['associations'][j]['object']['label']
            else:
                curdata = gene_GO_dict[curlist[1]] 
                if curdata.find(getdata['associations'][j]['object']['label']) == -1:
                    gene_GO_dict[curlist[1]] = curdata+", "+getdata['associations'][j]['object']['label']
 

my_html = '<table><thead><tr><th>Gene name</th><th>GO term</th></tr></thead><tbody>{}</tbody></table>'
rows = []

for k, v in gene_GO_dict.items():
    rows.append('<tr><td>{}</td><td>{}</td></tr>'.format(k.decode("utf-8") , v))
result = my_html.format(''.join(rows))
display(HTML(result))

FA_effector_list= FA_effector[0].splitlines()
#print( len(FA_core_list))

gene_GO_dict_effectors = dict()

for i in range(0, len(FA_effector_list)):
    curlist = FA_effector_list[i].split()
    curid = curlist[0].decode("utf-8") 
    print(curid)
        
    qurl = "https://api.monarchinitiative.org/api/bioentity/gene/"+curid+"/function/"
    print(qurl)
    with urllib.request.urlopen(qurl) as thisurl:
        getdata = json.loads(thisurl.read().decode())

        for j in range(0, len(getdata['associations'])):
            #print(getdata['associations'][j]['object']['id'])
            if curlist[1] not in gene_GO_dict:
                gene_GO_dict_effectors[curlist[1]] = getdata['associations'][j]['object']['label']
            else:
                curdata = gene_GO_dict_effectors[curlist[1]] 
                if curdata.find(getdata['associations'][j]['object']['label']) == -1:
                    gene_GO_dict_effectors[curlist[1]] = curdata+", "+getdata['associations'][j]['object']['label']

my_html = '<table><thead><tr><th>Gene name</th><th>GO term</th></tr></thead><tbody>{}</tbody></table>'
rows = []

for k, v in gene_GO_dict_effectors.items():
    rows.append('<tr><td>{}</td><td>{}</td></tr>'.format(k.decode("utf-8") , v))
result = my_html.format(''.join(rows))
display(HTML(result))

FA_associated_list= FA_assoc[0].splitlines()
#print( len(FA_core_list))

gene_GO_dict_associated = dict()

for i in range(0, len(FA_associated_list)):
    curlist = FA_associated_list[i].split()
    curid = curlist[0].decode("utf-8") 
    print(curid)
        
    qurl = "https://api.monarchinitiative.org/api/bioentity/gene/"+curid+"/function/"
    print(qurl)
    with urllib.request.urlopen(qurl) as thisurl:
        getdata = json.loads(thisurl.read().decode())

        for j in range(0, len(getdata['associations'])):
            #print(getdata['associations'][j]['object']['id'])
            if curlist[1] not in gene_GO_dict:
                gene_GO_dict_associated[curlist[1]] = getdata['associations'][j]['object']['label']
            else:
                curdata = gene_GO_dict_associated[curlist[1]] 
                if curdata.find(getdata['associations'][j]['object']['label']) == -1:
                    gene_GO_dict_associated[curlist[1]] = curdata+", "+getdata['associations'][j]['object']['label']

my_html = '<table><thead><tr><th>Gene name</th><th>GO term</th></tr></thead><tbody>{}</tbody></table>'
rows = []

for k, v in gene_GO_dict_associated.items():
    rows.append('<tr><td>{}</td><td>{}</td></tr>'.format(k.decode("utf-8") , v))
result = my_html.format(''.join(rows))
display(HTML(result))

