get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import os
import re
from Bio import SeqIO
from Bio import SeqUtils
import pysam
from Bio.SeqRecord import SeqRecord
from pybedtools import BedTool
import numpy as np
import pybedtools
import time
import matplotlib.pyplot as plt
import sys
import subprocess
import shutil
from Bio.Seq import Seq
import pysam
from Bio import SearchIO
import json
import glob
import scipy.stats as stats
import statsmodels as sms
import statsmodels.sandbox.stats.multicomp
import distance
import seaborn as sns

#Define some PATH
BASE_AA_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12'
POST_ALLELE_ANALYSIS_PATH = os.path.join(BASE_AA_PATH, 'post_allele_analysis',                 'proteinortho_graph516_QC_Qcov80_PctID70_evalue01')
OUT_PATH = os.path.join(POST_ALLELE_ANALYSIS_PATH ,                        'secretome_expression_clusters')
CLUSTER_PATH_P = os.path.join(BASE_AA_PATH, 'Pst_104E_genome',                              'gene_expression', 'Pst104_p_SecretomeClustering' )
CLUSTER_PATH_H = os.path.join(BASE_AA_PATH, 'Pst_104E_genome',                              'gene_expression', 'Pst104_h_SecretomeClustering' )

#some list to order the output later on
haplotig_cluster_order = ['Cluster9', 'Cluster10', 'Cluster11', 'Cluster12', 'Cluster13', 'Cluster14',        'Cluster15', 'Cluster16']
primary_cluster_order = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6',        'Cluster7', 'Cluster8']

#get the different classes of genes e.g. alleles, non-allelic protein 'orthoglos',  \
#loose_singletons (still including unphased genes), singletons
allele_fn = 'Pst_104E_v12_p_ctg.all.alleles'
loose_singletons_fn = 'Pst_104E_v12_ph_ctg.loose_singletons'
singletons_fn = 'Pst_104E_v12_ph_ctg.singletons'
nap_fn = 'Pst_104E_v12_ph_ctg.no_alleles_orthologs'

alleles_df = pd.read_csv(os.path.join(POST_ALLELE_ANALYSIS_PATH, allele_fn), header=None,                        sep='\t', names=['p_genes', 'h_genes'])
loose_sing_array = pd.read_csv(os.path.join(POST_ALLELE_ANALYSIS_PATH, loose_singletons_fn),
                             header=None, sep='\t')[0]
sing_array = pd.read_csv(os.path.join(POST_ALLELE_ANALYSIS_PATH, singletons_fn),
                             header=None, sep='\t')[0]
nap_array = pd.read_csv(os.path.join(POST_ALLELE_ANALYSIS_PATH, nap_fn),
                             header=None, sep='\t')[0]

#now get the different gene clusters in a df with the following set up
#columns = gene, cluster, allele status, allele_ID
primary_df = pd.DataFrame(columns=['gene', 'cluster_ID', 'allele_state', 'allele_ID'])
haplotig_df = pd.DataFrame(columns=['gene', 'cluster_ID', 'allele_state', 'allele_ID'])

#get the genes and the cluster ID as fn in list of equal lenght to be used as gene and 
#cluster_ID columns
_gene_list = []
_cluster_list = []
for file in [x for x in os.listdir(CLUSTER_PATH_P) if x.endswith('_DEs.fasta')]:
    for seq in SeqIO.parse(open(os.path.join(CLUSTER_PATH_P,file), 'r'), 'fasta'):
        _gene_list.append(seq.id)
        _cluster_list.append(file.split('_')[0])
primary_df.gene = _gene_list
primary_df.cluster_ID = _cluster_list

#now populate the allele_state list by setting the value in the allele_state column
#nomenclatures are alleles, nap, loose_singletons (unphased singletons), singletons (True singletons)

primary_df.loc[               primary_df[primary_df.gene.isin(alleles_df.p_genes)].index,               'allele_state'] = "allelic"
primary_df.loc[               primary_df[primary_df.gene.isin(sing_array)].index,               'allele_state'] = 'singleton'
primary_df.loc[               primary_df[primary_df.gene.isin(nap_array)].index,               'allele_state'] = 'nap'

#now do the same thing for the haplotig sequences
#get the genes and the cluster ID as fn in list of equal lenght to be used as gene and 
#cluster_ID columns
_gene_list = []
_cluster_list = []
for file in [x for x in os.listdir(CLUSTER_PATH_H) if x.endswith('_DEs.fasta')]:
    for seq in SeqIO.parse(open(os.path.join(CLUSTER_PATH_H,file), 'r'), 'fasta'):
        _gene_list.append(seq.id)
        _cluster_list.append(file.split('_')[0])
haplotig_df.gene = _gene_list
haplotig_df.cluster_ID = _cluster_list
haplotig_df.loc[               haplotig_df[haplotig_df.gene.isin(alleles_df.h_genes)].index,               'allele_state'] = "allelic"
haplotig_df.loc[               haplotig_df[haplotig_df.gene.isin(loose_sing_array)].index,               'allele_state'] = 'singleton'
haplotig_df.loc[               haplotig_df[haplotig_df.gene.isin(nap_array)].index,               'allele_state'] = 'nap'

#now summarize the allele states and write them out to file
#first aggregateon cluster_ID and allele_state + unstack
primary_allele_state_df = primary_df.loc[:,['gene','cluster_ID','allele_state']].pivot_table(columns=['cluster_ID','allele_state'],aggfunc='count').unstack()
#drop the unneccessary gene level from the index and replace na with 0
primary_allele_state_df.index = primary_allele_state_df.index.droplevel()
primary_allele_state_df.fillna(0)
#add a total number as well
primary_allele_state_df['Total'] = primary_allele_state_df.sum(axis=1)
#save dataframe
out_fn = 'Pst_104E_v12_p_ctg.cluster_status_summary.df'
primary_allele_state_df.fillna(0).T.loc[:,        ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6',        'Cluster7', 'Cluster8']].to_csv(os.path.join(OUT_PATH, out_fn), sep='\t')

#now summarize the allele states and write them out to file
#first aggregateon cluster_ID and allele_state + unstack
haplotig_allele_state_df = haplotig_df.loc[:,['gene','cluster_ID','allele_state']].pivot_table(columns=['cluster_ID','allele_state'],aggfunc='count').unstack()
#drop the unneccessary gene level from the index and replace na with 0
haplotig_allele_state_df.index = haplotig_allele_state_df.index.droplevel()
haplotig_allele_state_df.fillna(0)
#add a total number as well
haplotig_allele_state_df['Total'] = haplotig_allele_state_df.sum(axis=1)
#save dataframe
out_fn = 'Pst_104E_v12_h_ctg.cluster_status_summary.df'
haplotig_allele_state_df.fillna(0).T.loc[:,        ['Cluster9', 'Cluster10', 'Cluster11', 'Cluster12', 'Cluster13', 'Cluster14',        'Cluster15', 'Cluster16']].to_csv(os.path.join(OUT_PATH, out_fn), sep='\t')

#get the allele for each gene using a dict approach that also takes care of potential multiple
# alleles
allele_single_dict = {}
allele_multiple_dict = {}
#take all the allelic genes and pick the corresponding allele form the allele_df
#if there are multiple possible allele pairings add those as list to a different dictionary
for gene in primary_df[primary_df.allele_state == 'allelic'].gene:
    if len(alleles_df[alleles_df.p_genes == gene].h_genes.tolist()) == 1:
        allele_single_dict[gene] = alleles_df[alleles_df.p_genes == gene].h_genes.tolist()[0]
    elif len(alleles_df[alleles_df.p_genes == gene].h_genes.tolist()) != 1:
        print(len(alleles_df[alleles_df.p_genes == gene].h_genes.tolist()))
        allele_multiple_dict[gene] = alleles_df[alleles_df.p_genes == gene].h_genes.tolist()

for gene in haplotig_df[haplotig_df.allele_state == 'allelic'].gene:
    if len(alleles_df[alleles_df.h_genes == gene].p_genes.tolist()) == 1:
        allele_single_dict[gene] = alleles_df[alleles_df.h_genes == gene].p_genes.tolist()[0]
    elif len(alleles_df[alleles_df.h_genes == gene].p_genes.tolist()) != 1:
        print(len(alleles_df[alleles_df.h_genes == gene].p_genes.tolist()))
        allele_multiple_dict[gene] = alleles_df[alleles_df.h_genes == gene].p_genes.tolist()

#frist add the single allele pairing to the dataframes
def add_single_alleles(x, _dict1=allele_single_dict,_dict2=allele_multiple_dict):
    if x in _dict1.keys():
        return _dict1[x]
    elif x in _dict2:
        return 'multiples'
        

primary_df.allele_ID = primary_df.gene.apply(add_single_alleles)
haplotig_df.allele_ID = haplotig_df.gene.apply(add_single_alleles)

#now take care of the genes that have multiple alleles. In our case the biggest possible number
#is two AND all are two so this hack

#make two copies of the df that are multiples
tmp0_df = primary_df[primary_df.allele_ID == 'multiples'].copy()
tmp1_df = primary_df[primary_df.allele_ID == 'multiples'].copy()
drop_index = primary_df[primary_df.allele_ID == 'multiples'].index
#add the genes ideas to each of the copies once taking the first element and the other time
#the second
tmp0_df.allele_ID = tmp0_df.gene.apply(lambda x: allele_multiple_dict[x][0])
tmp1_df.allele_ID = tmp1_df.gene.apply(lambda x: allele_multiple_dict[x][1])
#now concat both tmp dataframes to the original dataframe while not including them in the
#former
primary_wa_df = pd.concat([primary_df.drop(primary_df.index[drop_index]), tmp0_df, tmp1_df], axis = 0)
primary_wa_df.reset_index(drop=True, inplace=True)

#now take care of the genes that have multiple alleles. In our case the biggest possible number
#is two AND all are two so this hack

#make two copies of the df that are multiples
tmp0_df = haplotig_df[haplotig_df.allele_ID == 'multiples'].copy()
tmp1_df = haplotig_df[haplotig_df.allele_ID == 'multiples'].copy()
drop_index = haplotig_df[haplotig_df.allele_ID == 'multiples'].index
#add the genes ideas to each of the copies once taking the first element and the other time
#the second
tmp0_df.allele_ID = tmp0_df.gene.apply(lambda x: allele_multiple_dict[x][0])
tmp1_df.allele_ID = tmp1_df.gene.apply(lambda x: allele_multiple_dict[x][1])
#now concat both tmp dataframes to the original dataframe while not including them in the
#former
haplotig_wa_df = pd.concat([haplotig_df.drop(haplotig_df.index[drop_index]), tmp0_df, tmp1_df], axis = 0)
haplotig_wa_df.reset_index(drop=True, inplace=True)

#now summaries the respective cluster hits for primary contigs
count_list = []
percentage_list = []
for cluster in primary_df.cluster_ID.unique():
    c_genes = ''
    #subset the dataframe to get the allelic genes in each cluster
    c_genes = primary_df[(primary_df.cluster_ID == cluster)                          & (primary_df.allele_state == 'allelic')].gene
    #use this list to subset the other dataframe
    _tmp_df = haplotig_wa_df[haplotig_wa_df.allele_ID.isin(c_genes)]
    _tmp_df.rename(columns={'gene': cluster}, inplace=True)
    #count occurances and add them to the list to make a dataframe alter
    count_list.append(_tmp_df.groupby('cluster_ID').count()[cluster])
    #now take care of percentage by making a count dataframe 
    _tmp_count_df = _tmp_df.groupby('cluster_ID').count().copy()
    #and dividing series by the clusters total
    _tmp_count_df[cluster] = _tmp_count_df[cluster].        apply(lambda x: x/primary_allele_state_df.loc[cluster, "allelic"]*100)
    percentage_list.append(_tmp_count_df[cluster])

#now generate some summary df by concaonating the list and adding a Total line at     
c_out_fn = 'Pst_104E_v12_p_ctg.relatvie_cluster_allele_status_count_summary.df'
count_df = pd.concat(count_list, axis=1)
count_df.loc['Total',:]= count_df.sum(axis=0)
count_df.fillna(0, inplace=True)
count_df.astype(int).loc[haplotig_cluster_order+["Total"], primary_cluster_order]    .to_csv(os.path.join(OUT_PATH, c_out_fn), sep='\t')



p_out_fn = 'Pst_104E_v12_p_ctg.relatvie_cluster_allele_status_per_summary.df'
percentage_df = pd.concat(percentage_list, axis=1)
percentage_df.loc['Total',:]= percentage_df.sum(axis=0)
percentage_df.fillna(0, inplace=True)
percentage_df.round(1).loc[haplotig_cluster_order+["Total"], primary_cluster_order]    .to_csv(os.path.join(OUT_PATH, p_out_fn), sep='\t')

#now summaries the respective cluster hits for haplotigs
count_list = []
percentage_list = []
for cluster in haplotig_df.cluster_ID.unique():
    c_genes = ''
    #subset the dataframe to get the allelic genes in each cluster
    c_genes = haplotig_df[(haplotig_df.cluster_ID == cluster)                          & (haplotig_df.allele_state == 'allelic')].gene
    #use this list to subset the other dataframe
    _tmp_df = primary_wa_df[primary_wa_df.allele_ID.isin(c_genes)]
    _tmp_df.rename(columns={'gene': cluster}, inplace=True)
    #count occurances and add them to the list to make a dataframe alter
    count_list.append(_tmp_df.groupby('cluster_ID').count()[cluster])
    #now take care of percentage by making a count dataframe 
    _tmp_count_df = _tmp_df.groupby('cluster_ID').count().copy()
    #and dividing series by the clusters total
    _tmp_count_df[cluster] = _tmp_count_df[cluster].        apply(lambda x: x/haplotig_allele_state_df.loc[cluster, "allelic"]*100)
    percentage_list.append(_tmp_count_df[cluster])

#now generate some summary df by concaonating the list and adding a Total line at     
c_out_fn = 'Pst_104E_v12_h_ctg.relatvie_cluster_allele_status_count_summary.df'
count_df = pd.concat(count_list, axis=1)
count_df.loc['Total',:]= count_df.sum(axis=0)
count_df.fillna(0, inplace=True)
count_df.astype(int).loc[primary_cluster_order+["Total"], haplotig_cluster_order]    .to_csv(os.path.join(OUT_PATH, c_out_fn), sep='\t')



p_out_fn = 'Pst_104E_v12_h_ctg.relatvie_cluster_allele_status_per_summary.df'
percentage_df = pd.concat(percentage_list, axis=1)
percentage_df.loc['Total',:]= percentage_df.sum(axis=0)
percentage_df.fillna(0, inplace=True)
percentage_df.round(1).loc[primary_cluster_order+["Total"], haplotig_cluster_order]    .to_csv(os.path.join(OUT_PATH, p_out_fn), sep='\t')

#at the end fix up the allele summary dataframe for primary allele state analysis
#at this point we count the non-phased singletons to the alleles as well in the primary
#but leave them out initially for the relative analysis
reset_index = primary_df[(primary_df.allele_state != 'allelic')&(primary_df.allele_state != 'nap')          &(primary_df.allele_state != 'singleton')].index
primary_df.loc[reset_index, 'allele_state'] = 'allelic'
#save dataframe
#now summarize the allele states and write them out to file
#first aggregateon cluster_ID and allele_state + unstack
primary_allele_state_df = primary_df.loc[:,['gene','cluster_ID','allele_state']].pivot_table(columns=['cluster_ID','allele_state'],aggfunc='count').unstack()
#drop the unneccessary gene level from the index and replace na with 0
primary_allele_state_df.index = primary_allele_state_df.index.droplevel()
primary_allele_state_df.fillna(0)
#add a total number as well
primary_allele_state_df['Total'] = primary_allele_state_df.sum(axis=1)

out_fn = 'Pst_104E_v12_p_ctg.cluster_status_summary.df'
primary_allele_state_df.fillna(0).T.loc[:,        ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6',        'Cluster7', 'Cluster8']].to_csv(os.path.join(OUT_PATH, out_fn), sep='\t')

