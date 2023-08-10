get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import os
import re
from Bio import SeqIO
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

#Define the PATH
BASE_AA_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12'
BASE_A_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly'
ASSEMBLETICS_INPATH = os.path.join(BASE_AA_PATH, 'Assembletics')
BLAST_DB = os.path.join(BASE_AA_PATH, 'blast_DB')
OUT_PATH = os.path.join(BASE_AA_PATH, 'allele_analysis')
OUT_tmp = os.path.join(OUT_PATH, 'tmp')
PROTEINORTHO = os.path.join(BASE_AA_PATH, 'proteinortho')
#renamed the allele path to reflect the proteinortho results
ALLELE_PATH = os.path.join(OUT_PATH, 'alleles_proteinortho_graph516')
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)
if not os.path.isdir(OUT_tmp):
    os.mkdir(OUT_tmp)
if not os.path.exists(ALLELE_PATH):
    os.mkdir(ALLELE_PATH)

#Define your p and h genome and move it into the allele analysis folder
p_genome = 'Pst_104E_v12_p_ctg'
h_genome = 'Pst_104E_v12_h_ctg'
genome_file_suffix = '.genome_file'
for x in (x + '.fa' for x in [p_genome, h_genome]):
    shutil.copy2(BASE_A_PATH+'/'+x, OUT_tmp)

#get protetinortho syntany file which ends with .poff
poff_graph_fn = os.path.join(PROTEINORTHO, 'ph_ctg_516.poff-graph')

#Define ENV parameters for blast hits and threads used in blast analysis
n_threads = 16
e_value = 1e-3
Qcov_cut_off = 80 #this defines the mimimum coverage of the Query to be required for filtering. Will become part of name.
PctID_cut_off = 70 #this defines the mimimum PctID accross the alignment to be required for filtering. Will become part of name.

def att_column_p(x, y, z, a, b):
    '''Generate attribute column of h on p p_gff dataframe.
    '''
    string = 'h_contig=%s;query_start=%s;query_stop=%s;query_length=%s;query_aln_ln=%s' % (x, str(y), str(z), str(a), str(b))
    return string

def att_column_h(x, y, z, a, b):
    '''Generate attribute column of h on p h_gff dataframe.
    '''
    string = 'p_contig=%s;ref_start=%s;ref_stop=%s;ref_length=%s;ref_aln_ln=%s' % (x, str(y), str(z), str(a), str(b))
    return string

def gene_contig_subset(df, contig, feature):
    '''Input the gff and subset the feature columne by gene.
        Return data frame subset by contig and gene'''
    tmp_df = df[(df[0].str.contains(contig)) & (df[2]==feature)]
    tmp_df.sort_values(3,inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)
    return tmp_df

def same_contig_blast(x,y):
    '''Function that checks if the blast hit in columne y is on the same contig as the the query sequence in
    column y.
    '''
    q_contig = re.search(r'[p|h]([a-z]*_[0-9]*)_?', x).group(1)
    if y != 'NaN':
        hit_contig = re.search(r'[p|h]([a-z]*_[0-9]*)_?', y).group(1)
    else:
        hit_contig = 'NaN'
    if q_contig == hit_contig:
        return True
    else:
        return False

def target_on_maped_haplotig(t_contig, h_contig_overlap):
    '''Simple function that checks if the target contig is in the list of h overlaping contigs'''
    if t_contig == False or h_contig_overlap == False:
        return False  
    else:
        if t_contig in h_contig_overlap:
            return True
        else:
            return False
        

#generate the blast databases if not already present
os.chdir(BLAST_DB)
blast_dir_content = os.listdir(BLAST_DB)
for x in blast_dir_content:
    if x.endswith('.fa') and ({os.path.isfile(x + e) for e in ['.psq', '.phr', '.pin'] } != {True}           and {os.path.isfile(x + e) for e in ['.nin', '.nhr', '.nsq'] } != {True} ):

        make_DB_options = ['-in']
        make_DB_options.append(x)
        make_DB_options.append('-dbtype')
        if 'protein' in x:
            make_DB_options.append('prot')
        else:
            make_DB_options.append('nucl')
        make_DB_command = 'makeblastdb %s' % ' '.join(make_DB_options)
        make_DB_stderr = subprocess.check_output(make_DB_command, shell=True, stderr=subprocess.STDOUT)
        print('%s is done!' % make_DB_command)
print("All databases generated and ready to go!")

blast_dict = {}
blast_dict['gene_fa'] = [x for x  in os.listdir(BLAST_DB) if x.endswith('gene.fa') and 'ph_ctg' not in x]
blast_dict['protein_fa'] = [x for x  in os.listdir(BLAST_DB) if x.endswith('protein.fa') and 'ph_ctg' not in x]
blast_stderr_dict = {}
for key in blast_dict.keys():
    for n, query in enumerate(blast_dict[key]):
        tmp_list = blast_dict[key][:]
        tmp_stderr_list = []
        #this loops through all remaining files and does blast against all those
        del tmp_list[n]
        for db in tmp_list:
            blast_options = ['-query']
            blast_options.append(query)
            blast_options.append('-db')


            blast_options.append(db)
            blast_options.append('-outfmt 6')
            blast_options.append('-evalue')
            blast_options.append(str(e_value))
            blast_options.append('-num_threads')
            blast_options.append(str(n_threads))
            blast_options.append('>')
            if 'gene' in query and 'gene' in db:
                out_name_list = [ query.split('.')[0], db.split('.')[0], str(e_value), 'blastn.outfmt6']
                out_name = os.path.join(OUT_PATH,'.'.join(out_name_list))
                blast_options.append(out_name)
                blast_command = 'blastn %s' % ' '.join(blast_options)
            elif 'protein' in query and 'protein' in db:
                out_name_list = [ query.split('.')[0], db.split('.')[0], str(e_value), 'blastp.outfmt6']
                out_name = os.path.join(OUT_PATH,'.'.join(out_name_list))
                blast_options.append(out_name)
                blast_command = 'blastp %s' % ' '.join(blast_options)
            print(blast_command)
            if not os.path.exists(out_name):
                blast_stderr_dict[blast_command] = subprocess.check_output(blast_command, shell=True, stderr=subprocess.STDOUT)
                print("New blast run and done!")
            else:
                blast_stderr_dict[blast_command] = 'Previously done already!'
                print('Previously done already!')

#make a sequnece length dict for all genes and proteins for which blast was performed
seq_list = []
length_list =[]
for key in blast_dict.keys():
    for file in blast_dict[key]:
        for seq in SeqIO.parse(open(file), 'fasta'):
            seq_list.append(seq.id)
            length_list.append(len(seq.seq))
length_dict = dict(zip(seq_list, length_list))
        

#get assembletics folders
ass_folders = [os.path.join(ASSEMBLETICS_INPATH, x) for x in os.listdir(ASSEMBLETICS_INPATH) if x.endswith('_php_8kbp')]
ass_folders.sort()

#read in the gff files
p_gff = pd.read_csv(BASE_A_PATH+'/'+p_genome+'.anno.gff3', sep='\t', header=None)
h_gff = pd.read_csv(BASE_A_PATH+'/'+h_genome+'.anno.gff3', sep='\t', header=None)

#read in the blast df and add QLgth and QCov columns
blast_out_dict = {}
blast_header = ['Query', 'Target', 'PctID', 'AlnLgth', 'NumMis', 'NumGap', 'StartQuery', 'StopQuery', 'StartTarget',              'StopTarget', 'e-value','BitScore']
for key in blast_stderr_dict.keys():
    file = key.split('>')[-1][1:]
    tmp_key = file.split('/')[-1]
    tmp_df = pd.read_csv(file, sep='\t', header=None, names=blast_header)
    tmp_df["QLgth"] = tmp_df["Query"].apply(lambda x: length_dict[x])
    tmp_df["QCov"] = tmp_df['AlnLgth']/tmp_df['QLgth']*100
    tmp_df.sort_values(by=['Query', 'e-value','BitScore', ],ascending=[True, True, False], inplace=True)
    #now make sure to add proteins/genes without blast hit to the dataframes
    #check if gene blast or protein blast and pull out respective query file
    if file.split('.')[-2] == 'blastp':
        tmp_blast_seq = [x for x in blast_dict['protein_fa'] if x.startswith(tmp_key.split('.')[0])][0]
    elif file.split('.')[-2] == 'blastn':
        tmp_blast_seq = [x for x in blast_dict['gene_fa'] if x.startswith(tmp_key.split('.')[0])][0]
    tmp_all_blast_seq = []
    #make list off all query sequences
    for seq in SeqIO.parse(os.path.join(BLAST_DB, tmp_blast_seq), 'fasta'):
        tmp_all_blast_seq.append(str(seq.id))
    tmp_all_queries_w_hit = tmp_df["Query"].unique()
    tmp_queries_no_hit = set(tmp_all_blast_seq) - set(tmp_all_queries_w_hit)
    no_hit_list = []
    #loop over the quieres with no hit and make list of list out of them the first element being the query id
    for x in tmp_queries_no_hit:
        NA_list = ['NaN'] * len(tmp_df.columns)
        NA_list[0] = x
        no_hit_list.append(NA_list)
    tmp_no_hit_df = pd.DataFrame(no_hit_list)
    tmp_no_hit_df.columns = tmp_df.columns
    tmp_no_hit_df['QLgth'] = tmp_no_hit_df.Query.apply(lambda x: length_dict[x])
    tmp_df = tmp_df.append(tmp_no_hit_df)
    tmp_df['q_contig'] = tmp_df['Query'].str.extract(r'([p|h][a-z]*_[^.]*).?')
    tmp_df['t_contig'] = tmp_df['Target'].str.extract(r'([p|h][a-z]*_[^.]*).?')
    #fix that if you don't extract anything return False and not 'nan'
    tmp_df['t_contig'].fillna(False, inplace=True)
    tmp_df['q_contig == t_contig'] = tmp_df.apply(lambda row: same_contig_blast(row['Query'], row['Target']), axis =1)
    tmp_df.reset_index(inplace=True, drop=True)
    blast_out_dict[tmp_key] = tmp_df.iloc[:,:]
    

for folder in ass_folders:
    '''
    From oriented_coords.csv files generate gff files with the following set up.
    h_contig overlap
    'query', 'ref', 'h_feature', 'query_start', 'query_end', 'query_aln_len', 'strand', 'frame', 'attribute_h'
    p_contig overlap
    'ref', 'query', 'p_feature','ref_start','ref_end', 'alignment_length', 'strand', 'frame', 'attribute_p'.
    Save those file in a new folder for further downstream analysis. The same folder should include the contig and gene filtered
    gffs. 
    '''
    orient_coords_file = [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith('oriented_coords.csv')][0]
    #load in df and generate several additions columns
    tmp_df = pd.read_csv(orient_coords_file, sep=',')
    #check if there is any resonable alignment if not go to the next one
    if len(tmp_df) == 0:
        print('Check on %s assembletics' % (folder))
        continue
    tmp_df['p_feature'] = "haplotig"
    tmp_df['h_feature'] ='primary_contig'
    tmp_df['strand'] = "+"
    tmp_df['frame'] = 0
    tmp_df['query_aln_len'] = abs(tmp_df['query_end']-tmp_df['query_start'])
    tmp_df['alignment_length'] = abs(tmp_df['ref_end'] - tmp_df['ref_start'])
    tmp_df.reset_index(drop=True, inplace=True)
    tmp_df.sort_values('query', inplace=True)
    tmp_df['attribute_p'] = tmp_df.apply(lambda row: att_column_p(row['query'],row['query_start'], row['query_end'],                                                         row['query_length'], row['query_aln_len']), axis=1)
    
    tmp_df['attribute_h'] = tmp_df.apply(lambda row: att_column_h(row['ref'], row['ref_start'], row['ref_end'],                                                                  row['ref_length'], row['alignment_length']), axis=1)
    
    #generate tmp gff dataframe
    tmp_p_gff_df = tmp_df.loc[:, ['ref', 'query', 'p_feature','ref_start','ref_end', 'alignment_length', 'strand', 'frame', 'attribute_p']]
    #now filter added if start < stop swap round. And if start == 0. This was mostly?! only? the case for haplotigs
    tmp_p_gff_df['comp'] = tmp_p_gff_df['ref_start'] - tmp_p_gff_df['ref_end']
    tmp_swap_index  = tmp_p_gff_df['comp'] > 0
    tmp_p_gff_df.loc[tmp_swap_index, 'ref_start'] , tmp_p_gff_df.loc[tmp_swap_index, 'ref_end'] = tmp_p_gff_df['ref_end'], tmp_p_gff_df['ref_start']
    #and if 'ref_start' == g
    is_null_index  = tmp_p_gff_df['ref_start'] == 0
    tmp_p_gff_df.loc[is_null_index, 'ref_start'] = 1
    tmp_p_gff_df = tmp_p_gff_df.iloc[:, 0:9]
    tmp_p_gff_df.sort_values('ref_start',inplace=True)
    tmp_h_gff_df = tmp_df.loc[:, ['query', 'ref', 'h_feature', 'query_start', 'query_end', 'query_aln_len', 'strand', 'frame', 'attribute_h']]
    #same fix for tmp_h_gff
    tmp_h_gff_df['comp'] = tmp_h_gff_df['query_start'] - tmp_h_gff_df['query_end']
    tmp_swap_index  = tmp_h_gff_df['comp'] > 0
    tmp_h_gff_df.loc[tmp_swap_index, 'query_start'] , tmp_h_gff_df.loc[tmp_swap_index, 'query_end'] = tmp_h_gff_df['query_end'], tmp_h_gff_df['query_start']
    #and if 'query_start' == g
    is_null_index  = tmp_h_gff_df['query_start'] == 0
    tmp_h_gff_df.loc[is_null_index, 'query_start'] = 1
    tmp_h_gff_df = tmp_h_gff_df.iloc[:, 0:9]
    tmp_h_gff_df.sort_values('query_start', inplace=True)
    #make the outfolder and save gff files of overlap and filtered gene files
    folder_suffix = folder.split('/')[-1]
    #get the contig
    contig = re.search(r'p([a-z]*_[0-9]*)_', folder_suffix).group(1)
    out_folder = os.path.join(OUT_PATH, folder_suffix)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_name_p = os.path.join(out_folder, folder_suffix )
    tmp_p_gff_df.to_csv(out_name_p+'.p_by_h_cov.gff', sep='\t' ,header=None, index =None)
    out_name_h =  os.path.join(out_folder, folder_suffix)
    tmp_h_gff_df.to_csv(out_name_h+'.h_by_p_cov.gff' , sep='\t' ,header=None, index =None)
    #write those out to new folder for php together with the p and h gff of genes
    p_gene_gff = gene_contig_subset(p_gff, contig, 'gene')
    p_gene_gff.to_csv(out_name_p+'.p_gene.gff', sep='\t' ,header=None, index =None)
    h_gene_gff = gene_contig_subset(h_gff, contig, 'gene')
    h_gene_gff.to_csv(out_name_h+'.h_gene.gff', sep='\t' ,header=None, index =None)
    
    #next would be to do overlap between the gene gff and the corresponding alingment gff. Write this out. Dict for each gene and its corresponding h_contig/p_contig
    

#now loop over the outfolders
out_folders = [os.path.join(OUT_PATH, x) for x in os.listdir(OUT_PATH) if x.endswith('_php_8kbp')]
out_folders.sort()

p_gene_h_contig_overlap_dict = {}
for out_folder in out_folders:
    tmp_folder_content = os.listdir(out_folder)
    tmp_p_gene_gff = [os.path.join(out_folder, x) for x in tmp_folder_content if x.endswith('p_gene.gff') ][0]
    tmp_p_gene_bed = pybedtools.BedTool(tmp_p_gene_gff).remove_invalid().saveas()
    tmp_p_by_h_gff = [os.path.join(out_folder, x) for x in tmp_folder_content if x.endswith('p_by_h_cov.gff') ][0]
    tmp_p_by_h_bed = pybedtools.BedTool(tmp_p_by_h_gff).remove_invalid().saveas()
    #generate a bed intersect
    p_gene_and_p_by_h = tmp_p_gene_bed.intersect(tmp_p_by_h_bed, wb=True)
    #check the length of the obtained intersect is 
    print("This is the lenght of the intersect %i" %len(p_gene_and_p_by_h ))
    if len(p_gene_and_p_by_h) == 0:
        continue
    #turn this into a dataframe 
    p_gene_and_p_by_h_df = p_gene_and_p_by_h.to_dataframe()
    p_gene_and_p_by_h_df['p_gene'] = p_gene_and_p_by_h_df[8].str.extract(r'ID=([^;]*);')
    p_gene_and_p_by_h_df['p_protein'] = p_gene_and_p_by_h_df['p_gene'].str.replace('TU', 'model')
    #safe this in the same folder as dataframe
    p_gene_and_p_by_h_df.to_csv(tmp_p_gene_gff.replace('p_gene.gff', 'gene_haplotig_intersect.df'), sep='\t', index=None)
    p_gene_and_p_by_h_df_2 = p_gene_and_p_by_h_df.loc[:, [10, 'p_protein']]
    #make a dict for the overlap of each gene with it's correspoding haplotig
    tmp_p_gene_h_overlap_df = p_gene_and_p_by_h_df_2.groupby('p_protein')[10].apply(list)
    tmp_p_gene_h_overlap_dict = dict(zip(tmp_p_gene_h_overlap_df.index, tmp_p_gene_h_overlap_df.values))
    p_gene_h_contig_overlap_dict.update(tmp_p_gene_h_overlap_dict)
    

#get out the p_protein on h_protein blast out dataframe and appand the h_contig overlap for each protein
_key = [x for x in blast_out_dict.keys() if x.startswith(p_genome) and x.split('.')[-2] == 'blastp' and x.endswith('outfmt6')][0]
p_gene_h_contig_overlap_df = pd.DataFrame([p_gene_h_contig_overlap_dict.keys(), p_gene_h_contig_overlap_dict.values()],  index = ['p_protein', 'h_contig_overlap']).T
_tmp_df = blast_out_dict[_key]
_tmp_df = _tmp_df.merge(p_gene_h_contig_overlap_df, how='outer' ,left_on='Query', right_on='p_protein')
_tmp_df['h_contig_overlap'].fillna(False, inplace=True)
#check if the protein hit resites on the overlapping haplotig
_tmp_df['t_contig == h_contig_overlap'] = _tmp_df.apply(lambda row: target_on_maped_haplotig(row['t_contig'], row['h_contig_overlap']), axis =1 )
#blast_out_dict[_key] = _tmp_df

#write the dataframe again
_tmp_df.to_csv(os.path.join(OUT_PATH, _key+'.allele_analysis'), index=None, sep='\t')

#now mangale the data a bit
#write out t_contig == h_contig_overlap best hits

#Qcov_cut_off = ''
#PctID_cut_off = ''
if Qcov_cut_off and PctID_cut_off:
    pass
else: 
    print('No cut off defined! Do you want to define it. Please go ahead above.')

#now pull in the proteinortho_df for allele analysis
#this will add a new column to the dataframe called PO_allele which will be True when protein_ortho
#says it is 
#co-orthologs will be filtered based on same contig, and e-values

poff_graph_header = ['Target', 'Query', 'evalue_ab', 'bitscore_ab', 'evalue_ba', 'bitscore_ba', 'same_strand' , 'simscore']
po_df = pd.read_csv(poff_graph_fn, sep='\t', header=None, names=poff_graph_header, comment='#' )

#filter out the co-orthologs
#co_po_df = po_df[po_df.Target.str.contains(',')].loc[:, ['Target', 'Query']].copy()
#po_df = po_df[~po_df.Target.str.contains(',')].loc[:, ['Target', 'Query']]
#not required anymore as those are pairs in the graph file as well

#no add a comparision column to po_df and _tmp_df
po_df['comp'] = po_df['Query'] + po_df['Target']
_tmp_df['comp'] = _tmp_df['Query'] + _tmp_df['Target']
#generate a new PO_allele column
_tmp_df['PO_allele'] = False
#and set it to True
_tmp_df.loc[_tmp_df[_tmp_df.comp.isin(po_df.comp.unique())].index, 'PO_allele' ]= True
#now drop the comp column again
_tmp_df = _tmp_df.drop('comp', 1)

#this is changed in version two to account for PO_allele analysis as well
_tmp_grouped = _tmp_df[(_tmp_df['t_contig == h_contig_overlap'] == True)                      & (_tmp_df['PO_allele'] ==  True)].groupby('Query')

#filter on the evalue and than on the BitScore
best_hits_t_contig_and_h_contig = _tmp_grouped.apply(lambda g: g[(g['e-value'] == g['e-value'].min())])
best_hits_t_contig_and_h_contig = best_hits_t_contig_and_h_contig.reset_index(drop=True).groupby('Query').apply(lambda g: g[(g['BitScore'] == g['BitScore'].max())])

out_name = p_genome + '.h_contig_overlap.alleles'

best_hits_t_contig_and_h_contig.loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
#now save including the filter as well
if Qcov_cut_off and PctID_cut_off:
    out_name = p_genome + '.h_contig_overlap.Qcov%s.PctID%s.alleles' % (Qcov_cut_off,PctID_cut_off)
    best_hits_t_contig_and_h_contig[(best_hits_t_contig_and_h_contig['QCov'] >= Qcov_cut_off) & (best_hits_t_contig_and_h_contig['PctID'] >= PctID_cut_off)]    .loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
else:
    print("No filter selected for Qcov and PctID cut-off. If single filter desired set other filter to 1000.")

#test if filtering like this works
if set(_tmp_df[(_tmp_df['t_contig == h_contig_overlap'] == True)                      & (_tmp_df['PO_allele'] ==  True)]['Query'].unique()) == set(best_hits_t_contig_and_h_contig['Query'].unique().tolist()):
    print("All good please proceed!")
else:
    print('The filter threw an error. Not the expected output')

#now drop everything that was already captured
query_ids_to_drop = best_hits_t_contig_and_h_contig['Query'].unique().tolist()
target_ids_to_drop = best_hits_t_contig_and_h_contig['Target'].unique().tolist()

#get all queries that are on the respective haplotig but have not been caught before
_tmp_groupbed = _tmp_df[(~_tmp_df['Query'].isin(query_ids_to_drop))&                        (_tmp_df['q_contig == t_contig'] == True)&                        (_tmp_df['PO_allele'] ==  True)].groupby('Query')
best_hits_same_contig_no_overlap = _tmp_groupbed.apply(lambda g: g[(g['e-value'] == g['e-value'].min())]) 
best_hits_same_contig_no_overlap = best_hits_same_contig_no_overlap.reset_index(drop=True).groupby('Query').apply(lambda g: g[(g['BitScore'] == g['BitScore'].max())])
out_name = p_genome + '.no_specific_h_contig_overlap.alleles'
best_hits_same_contig_no_overlap.loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)

#now save including the filter as well
if Qcov_cut_off and PctID_cut_off:
    out_name = p_genome + '.no_specific_h_contig_overlap.Qcov%s.PctID%s.alleles' % (Qcov_cut_off,PctID_cut_off)
    best_hits_same_contig_no_overlap[(best_hits_same_contig_no_overlap['QCov'] >= Qcov_cut_off) & (best_hits_same_contig_no_overlap['PctID'] >= PctID_cut_off)]    .loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
else:
    print("No filter selected for Qcov and PctID cut-off. If single filter desired set other filter to 1000.")

#test if filtering like this works
if set(_tmp_df[(~_tmp_df['Query'].isin(query_ids_to_drop))& (_tmp_df['q_contig == t_contig'] == True)              & (_tmp_df['PO_allele'] ==  True)]['Query'].unique())        == set(best_hits_same_contig_no_overlap['Query'].unique().tolist()):
    print("All good please proceed!")
else:
    print('The filter threw an error. Not the expected output')

#now drop everything that was already captured
query_ids_to_drop = query_ids_to_drop + best_hits_same_contig_no_overlap['Query'].unique().tolist()
target_ids_to_drop = target_ids_to_drop + best_hits_same_contig_no_overlap['Target'].unique().tolist()

#get all else. Meaning not on the corresponding haplotig but other haplotig

_tmp_groupbed = _tmp_df[(~_tmp_df['Query'].isin(query_ids_to_drop))&                        (_tmp_df['q_contig == t_contig'] == False)                        & (_tmp_df['Target'] != 'NaN' )&
                        (_tmp_df['PO_allele'] ==  True)].groupby('Query')
best_hits_different_contig = _tmp_groupbed.apply(lambda g: g[(g['e-value'] == g['e-value'].min()) ])
best_hits_different_contig = best_hits_different_contig .reset_index(drop=True).groupby('Query').apply(lambda g: g[(g['BitScore'] == g['BitScore'].max())])
out_name = p_genome + '.no_respective_h_contig_overlap.alleles'
best_hits_different_contig.loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)

#now save including the filter as well
if Qcov_cut_off and PctID_cut_off:
    out_name = p_genome + '.no_respective_h_contig_overlap.Qcov%s.PctID%s.alleles' % (Qcov_cut_off,PctID_cut_off)
    best_hits_different_contig[(best_hits_different_contig['QCov'] >= Qcov_cut_off) & (best_hits_different_contig['PctID'] >= PctID_cut_off)]    .loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
else:
    print("No filter selected for Qcov and PctID cut-off. If single filter desired set other filter to 1000.")

if set(_tmp_df[(~_tmp_df['Query'].isin(query_ids_to_drop))& (_tmp_df['q_contig == t_contig'] == False) &               (_tmp_df['Target'] != 'NaN' )              &(_tmp_df['PO_allele'] ==  True)]['Query'].unique().tolist())        == set(best_hits_different_contig['Query'].unique().tolist()):
    print("All good please proceed!")
else:
    print('The filter threw an error. Not the expected output')

#now drop everything that was already captured
query_ids_to_drop = query_ids_to_drop + best_hits_different_contig['Query'].unique().tolist()
target_ids_to_drop = target_ids_to_drop + best_hits_different_contig['Target'].unique().tolist()

#now print out the p_proteins without alleles
out_name = p_genome + '.no_alleles'
_tmp_df[~_tmp_df['Query'].isin(query_ids_to_drop)].drop_duplicates(subset='Query')['Query'].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)

#for filtering different query and targets to drop are required to be dropped
if Qcov_cut_off and PctID_cut_off:
    query_ids_to_drop_filtered =     best_hits_t_contig_and_h_contig[(best_hits_t_contig_and_h_contig['QCov'] >= Qcov_cut_off) & (best_hits_t_contig_and_h_contig['PctID']>= PctID_cut_off)]['Query'].unique().tolist()     + best_hits_same_contig_no_overlap[(best_hits_same_contig_no_overlap['QCov'] >= Qcov_cut_off) & (best_hits_same_contig_no_overlap['PctID'] >= PctID_cut_off)]['Query'].unique().tolist()    + best_hits_different_contig[(best_hits_different_contig['QCov'] >= Qcov_cut_off) & (best_hits_different_contig['PctID'] >= PctID_cut_off)]['Query'].unique().tolist()
    
    target_ids_to_drop_filtered =     best_hits_t_contig_and_h_contig[(best_hits_t_contig_and_h_contig['QCov'] >= Qcov_cut_off) & (best_hits_t_contig_and_h_contig['PctID']>= PctID_cut_off)]['Target'].unique().tolist()    + best_hits_same_contig_no_overlap[(best_hits_same_contig_no_overlap['QCov'] >= Qcov_cut_off) & (best_hits_same_contig_no_overlap['PctID'] >= PctID_cut_off)]['Target'].unique().tolist()     + best_hits_different_contig[(best_hits_different_contig['QCov'] >= Qcov_cut_off) & (best_hits_different_contig['PctID'] >= PctID_cut_off)]['Target'].unique().tolist()
    out_name = p_genome + '.Qcov%s.PctID%s.no_alleles' % (Qcov_cut_off,PctID_cut_off)
    #save the dataframe
    _tmp_df[~_tmp_df['Query'].isin(query_ids_to_drop_filtered)].drop_duplicates(subset='Query')['Query'].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)    
else:
    print("No filter selected for Qcov and PctID cut-off. If single filter desired set other filter to 1000.")                                                                                                                             

#save back the dataframe with '.analyzed' added to the key
blast_out_dict[_key+'.analyzed'] = _tmp_df

#pull out the h on p blast dataframe
_key = [x for x in blast_out_dict.keys() if x.startswith(h_genome) and x.split('.')[-2] == 'blastp' and x.endswith('outfmt6')][0]
_tmp_df = blast_out_dict[_key]

#pull out all h_contigs without alleles
_tmp_df_no_alleles = _tmp_df[~_tmp_df['Query'].isin(target_ids_to_drop)]
out_name = h_genome + '.no_alleles'
_tmp_df_no_alleles.drop_duplicates(subset='Query')['Query'].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)

#now pull out all h_proteins that have a blast hit but were left behind with selection 
#based on proteinortho and best blast hits above. No QCov and PctID filtering
_tmp_df_no_alleles_grouped = _tmp_df_no_alleles[_tmp_df_no_alleles["Target"] != 'NaN'].groupby('Query')
h_protein_no_alleles = _tmp_df_no_alleles_grouped.apply(lambda g: g[(g['e-value'] == g['e-value'].min())]) 
h_protein_no_alleles = h_protein_no_alleles.reset_index(drop=True).groupby('Query').apply(lambda g: g[(g['BitScore'] == g['BitScore'].max())])
out_name = h_genome + '.best_p_hits.no_alleles'
h_protein_no_alleles.loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
#write done the hits with no blast hit at all. Meaning blasting h on p gave no results
out_name = h_genome + '.no_p_hits.no_alleles'
_tmp_df_no_alleles[_tmp_df_no_alleles["Target"] == 'NaN']['Query'].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None) 

#now pull out all h_proteins that have a blast hit but were left behind with selection of best blast hits above. With QCov and PctID filtering.
if Qcov_cut_off and PctID_cut_off:
#pull out all h proteins without alleles when filtering was applied
    _tmp_df_no_alleles_filtered = _tmp_df[~_tmp_df['Query'].isin(target_ids_to_drop_filtered)]
    out_name = h_genome + '.Qcov%s.PctID%s.no_alleles' % (Qcov_cut_off,PctID_cut_off)
    _tmp_df_no_alleles_filtered.drop_duplicates(subset='Query')['Query'].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)

#pull out all the h proteins with blast hit when filtering was applied and filter those as well.
    #first get ride of all NAs in future version maybe convert QCov and PctID to floats or ints so this is not neccessary anymore
    _tmp_df_no_alleles_filtered_no_NA = _tmp_df_no_alleles_filtered[(_tmp_df_no_alleles_filtered["Target"] != 'NaN')]
    _tmp_df_no_alleles_filtered_grouped =         _tmp_df_no_alleles_filtered_no_NA[(_tmp_df_no_alleles_filtered_no_NA['QCov'] >= Qcov_cut_off) &                                    (_tmp_df_no_alleles_filtered_no_NA['PctID'] >= PctID_cut_off)].groupby('Query') 

    h_protein_no_alleles_filtered = _tmp_df_no_alleles_filtered_grouped.apply(lambda g: g[(g['e-value'] == g['e-value'].min())]) 
    h_protein_no_alleles_filtered = h_protein_no_alleles_filtered.reset_index(drop=True).groupby('Query').apply(lambda g: g[(g['BitScore'] == g['BitScore'].max())])
    out_name = h_genome + '.best_p_hits.Qcov%s.PctID%s.no_alleles' % (Qcov_cut_off,PctID_cut_off)
    h_protein_no_alleles_filtered.loc[:, ['Query', 'Target']].to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
    target_ids_to_drop_filtered = target_ids_to_drop_filtered + h_protein_no_alleles_filtered['Query'].unique().tolist()
    out_name = h_genome + '.no_p_hits.Qcov%s.PctID%s.no_alleles' % (Qcov_cut_off,PctID_cut_off)
    _tmp_df_no_alleles_filtered[~_tmp_df_no_alleles_filtered['Query'].isin(target_ids_to_drop_filtered)].drop_duplicates(subset='Query')['Query']    .to_csv(os.path.join(ALLELE_PATH, out_name), sep='\t', index=None, header=None)
else:
    print("No filter selected for Qcov and PctID cut-off. If single filter desired set other filter to 1000.")

#write the dataframe again
_tmp_df.to_csv(os.path.join(OUT_PATH, _key+'.allele_analysis'), index=None, sep='\t')

_tmp_df.columns

#might be good to write these analyzed dataframes out as well to make life a bit easier
blast_out_dict['Pst_104E_v12_p_ctg.Pst_104E_v12_h_ctg.0.001.blastp.outfmt6.analyzed'].columns





