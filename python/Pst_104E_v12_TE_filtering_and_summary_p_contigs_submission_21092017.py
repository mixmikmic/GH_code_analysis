get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os
import re
from Bio import SeqIO
import pysam
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SearchIO
from pybedtools import BedTool
import numpy as np
import pybedtools
import multiprocessing
import re
import time
import matplotlib.pyplot as plt

def ID_filter_gff(_feature, _id):
    """
    This filter parses out the top level id form the 9th gff column form a REPET gff file.
    It has a specific search pattern for each feature type in column 2.
    _type is defined by the feature '_'.join(feature.split("_")[-2:])
    This function expects that the variable genome either ends with p_ctg or h_ctg and adapts the
    search pattern accordingly.
    """
    _type = '_'.join(_feature.split("_")[-2:])
    if _type == 'REPET_TEs':
        if genome.endswith('p_ctg'):
            TE_pattern = r'ID=[A-Z,a-z,0-9,-]*_[A-Z,a-z,0-9]*_[0-9]*_([^;| ]*)'
        elif genome.endswith('h_ctg'):
            TE_pattern = r'ID=[A-Z,a-z,0-9,-]*_[A-Z,a-z,0-9]*_[0-9]*_[0-9]*_([^;| ]*)'
        TE_prog = re.compile(TE_pattern)
        TE_match = TE_prog.search(_id)

        try:
            return TE_match.group(1)
        except AttributeError:
            print(_id)

    if _type == 'REPET_SSRs':
        if genome.endswith('p_ctg'):
            SSR_pattern = 'ID=[A-Z,a-z,0-9,-]*_[A-Z,a-z,0-9]*_[0-9]*_([A-Z,a-z,0-9,-]*)'
        elif genome.endswith('h_ctg'):
            SSR_pattern = 'ID=[A-Z,a-z,0-9,-]*_[A-Z,a-z,0-9]*_[0-9]*_[0-9]*_([A-Z,a-z,0-9,-]*)'
        SSR_prog = re.compile(SSR_pattern)
        SSR_match = SSR_prog.search(_id)
        return SSR_match.group(1)
    if _type == 'REPET_tblastx' or _type == 'REPET_blastx':
        if genome.endswith('p_ctg'):
            blast_prog = re.compile(r'ID=[A-Z,a-z,0-9,-]*_[A-Z,a-z,0-9]*_[0-9]*_([^;| ]*)')
        elif genome.endswith('h_ctg'):
             blast_prog = re.compile(r'ID=[A-Z,a-z,0-9,-]*_[A-Z,a-z,0-9]*_[0-9]*_[0-9]*_([^;| ]*)')
        #blast_prog = re.compile(blast_pattern)
        blast_match = blast_prog.search(_id)
        return blast_match.group(1)

def blast_hit_gff(_feature, _row8, _id):
    """
    This filter parses the blast hit for REPET_TEs from the new 'ID' column. If no blast hit available returns Pastec ids.
    If the result is blast already the value is simple parse the blast hit.
    SSRs also get SSR
    !!!Requires the three_letter_dict to be defined previously.!!!
    _type is defined by the feature '_'.join(feature.split("_")[-2:])
    """
    _type = '_'.join(_feature.split("_")[-2:])
    if _type == 'REPET_TEs':
        #split the pastec_cat into the first three letter code
        #the spliting of the 'ID' column needs to be done differently depending on the h or p contigs.
        #h contigs contain one additional '_' in the contig id

        pastec_cat = _id.split('_')[0]
        if 'TE_BLR' in _row8:
            #hit_list = [x.split(';')[3] for x in _row8]
            blast_hit_pattern = r'TE_BLR\w*: (\S*)[ |;]'
            blast_hit_prog = re.compile(blast_hit_pattern)
            TE_match = blast_hit_prog.findall(_row8)
            first_sub_class = ':'.join(TE_match[0][:-1].split(':')[1:])
            if len([x for x in TE_match if first_sub_class in x]) == len(TE_match):
                if ';' in first_sub_class:
                    return first_sub_class.split(';')[0]
                else:
                    return first_sub_class
#fix this here to include the there letter code of the first bit of the ID similar to the blast hits
#e.g. ClassI:?:? and so on. a dict might be the easiest here.
            
            else:
                return three_letter_dict[pastec_cat]
        else:
            return three_letter_dict[pastec_cat]
    if _type == 'REPET_SSRs':
        return 'SSR'
        

        return SSR_match.group(1)
    if _type == 'REPET_tblastx' or _type == 'REPET_blastx':
        return ':'.join(_id.split(':')[1:])

def TE_classification_filter(_id, level = 0):
    """
    This function pulls out the class == level1, Order == level2, Superfamily == leve3.
    If SSR or noCat return these values.
    
    """
    if len(_id.split(':')) == 1:
        return _id
    if level == 0:
        _class = _id.split(':')[0]
        if _class == 'ClassI':
            return 'Retrotransposon'
        if _class == 'ClassII':
            return 'DNA_transposon'
    elif level == 1:
        _order = _id.split(':')[1]
        if _order == '?':
            return 'noCat'
        else:
            return _order
    elif level == 2:
        _superfamily = _id.split(':')[2]
        if _superfamily == '?':
            return 'noCat'
        else:
            return _superfamily
    else:
        print('Something wrong! Check if level is 0, 1 or 2')

source_dir = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly'


genome = 'Pst_104E_v12_p_ctg'

out_dir = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12/TE_analysis'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

#remove all commenting lines from the initial repet file
get_ipython().system('grep -v "^#" {source_dir}/{genome}.REPET.gff > {out_dir}/{genome}.REPET.gff')

p_repet_gff = pd.read_csv(out_dir+'/'+genome+'.REPET.gff', sep='\t', header = None)

#This needs to be updated here according to genome
TE_post_analysis_p = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/REPET/Pst79_p/Pst79_p_full_annotate/postanalysis/'

TE_post_analysis_p_header = 'TE      length  covg    frags   fullLgthFrags   copies  fullLgthCopies  meanId  sdId    minId   q25Id   medId   q75Id   maxId   meanLgth        sdLgth  minLgth q25Lgth medLgth q75Lgth maxLgth meanLgthPerc    sdLgthPerc      minLgthPerc  q25LgthPerc     medLgthPerc     q75LgthPerc     maxLgthPerc'.split(' ')

TE_post_analysis_p_header = [x for x in TE_post_analysis_p_header if x != '']

get_ipython().system('ls {TE_post_analysis_p}')

#this needs to be fixed up to pick the proper summary table
p_repet_summary_df = pd.read_csv(TE_post_analysis_p+'/'+'Pst79p_anno_chr_allTEs_nr_noSSR_join_path.annotStatsPerTE.tab' ,                                names = TE_post_analysis_p_header, header=None, sep='\t', skiprows=1 )

#check if I can filter the tab files for removing all TEs that are on the 2000 plus contigs
#remove tRNAs TEs with infernal

p_repet_summary_df['Code'] = p_repet_summary_df['TE'].apply(lambda x: x.split('_')[0])

code_keys = p_repet_summary_df['Code'].unique()

code_keys.sort()

code_long = ['DNA_transposon Helitron', 'DNA_transposon Helitron', 'DNA_transposon Helitron', 'DNA_transposon Maverick',            'DNA_transposon TIR', 'DNA_transposon TIR', 'DNA_transposon TIR', 'DNA_transposon TIR', 'DNA_transposon noCat',             'DNA_transposon MITE','DNA_transposon MITE', 'Potential Host Gene', 'Retrotransposon LINE', 'Retrotransposon LINE',             'Retrotransposon LINE','Retrotransposon LTR','Retrotransposon LTR', 'Retrotransposon LTR', 'Retrotransposon LTR', 'Retrotransposon PLE',              'Retrotransposon SINE',  'Retrotransposon SINE', 'Retrotransposon noCat', 'Retrotransposon LARD',             'Retrotransposon LARD', 'Retrotransposon TRIM', 'Retrotransposon TRIM', 'Retrotransposon noCat',               'Retrotransposon DIRS','Retrotransposon DIRS','Retrotransposon DIRS','Retrotransposon DIRS',             'noCat', 'noCat']
if len(code_keys) != len(code_long):
    print('Check the code_long list, because different length of keys and values!\n\n')
else:
    print('Check the code dict anyway')
code_dict = dict(zip(code_keys, code_long))
print(code_dict)

p_repet_summary_df['Code long'] = p_repet_summary_df['Code'].apply(lambda x: code_dict[x])
p_repet_summary_sum_df = pd.pivot_table(p_repet_summary_df, values=['covg', 'copies'], index='Code long', aggfunc=np.sum)
p_repet_summary_mean_df = pd.pivot_table(p_repet_summary_df, values='length', index='Code long', aggfunc=np.mean)
pd.concat([p_repet_summary_sum_df,p_repet_summary_mean_df], axis=1 )

#now filter the gff dataframe to delete all the high coverage contigs
#This might would have to be fixed as well. If we don't delete it as files should be already filtered
contigs_smaller_2000 = pd.read_csv('/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly/pcontig_smaller_2000.txt',                                  header=None)[0].tolist()

p_repet_gff = pd.read_csv(out_dir+'/'+genome+'.REPET.gff', sep='\t', header = None)

p_repet_gff_filtered = p_repet_gff[p_repet_gff[0].isin(contigs_smaller_2000)].reset_index(drop=True)

#filter out potential host genes
p_repet_gff_filtered = p_repet_gff_filtered[~p_repet_gff_filtered[8].str.contains("Potential")]

p_repet_gff_filtered['ID'] = p_repet_gff_filtered.apply(lambda row: ID_filter_gff(row[1], row[8]), axis=1)

#re-generate the code dict using the gff ID as Code keys


code_keys_gff = p_repet_gff_filtered[p_repet_gff_filtered[1].str.contains('REPET_TE')]['ID'].unique()

code_keys_gff = list({x.split('_')[0] for x in code_keys_gff})

code_keys_gff.sort()

#remove Potential host genes from long code list as those were filtered out previously.
code_long.remove('Potential Host Gene')

if len(code_keys_gff) != len(code_long):
    print("Go and check something is wrong at the code key stage!")

code_dict = dict(zip(code_keys_gff, code_long))

three_letter_code = list({x for x in code_keys_gff})

three_letter_code.sort()

three_letter_values = []
for x in three_letter_code:
    if 'MITE' in x:
        _value = "ClassII:MITE:?"
        three_letter_values.append(_value)
        continue
    if 'LARD' in x:
        _value = 'ClassI:LARD:?'
        three_letter_values.append(_value)
        continue
    if 'TRIM' in x:
        _value = 'ClassI:TRIM:?'
        three_letter_values.append(_value)
        continue
    _value =''
    if x[0] == 'D':
        _value = _value + 'ClassII:'
    if x[0] == 'R':
        _value = _value + 'ClassI:'
    if x[0] != 'D' and x[0] != 'R':
        _value = 'noCat'
        three_letter_values.append(_value)
        continue
    if x[1] == 'T':
        _value = _value + 'TIR:?'
    if x[1] == 'H':
        _value = _value + 'Helitron:?'
    if x[1] == 'M':
        _value = _value + 'Maverick:?'
    if x[0:2] == 'DY':
        _value = _value + ':Crypton:?'
    if x[1] == 'X':
        _value = _value + '?:?'
    if x[1] == 'I':
        _value = _value + 'LINE:?'
    if x[1] == 'L':
        _value = _value + 'LTR:?'
    if x[1] == 'P':
        _value = _value + 'Penelope:?'
    if x[1] == 'S':
        _value = _value + 'SINE:?'
    if x[0:2] == 'RY':
        _value = _value + 'DIRS:?'    
    three_letter_values.append(_value)

if len(three_letter_code) == len(three_letter_values):
    print("Aas")
    three_letter_dict = dict(zip(three_letter_code, three_letter_values))

three_letter_dict

p_repet_gff_filtered['Class:Order:Superfamily'] = p_repet_gff_filtered.apply(lambda row: blast_hit_gff(row[1], row[8], row['ID']), axis=1)

#generate a dict that can be used to rename the Class:Order:Superfamily column considering that partial matches ([2] == match_part) might contain different
#IDs even though they are the same TE only partial.
_tmp_subset = p_repet_gff_filtered[~p_repet_gff_filtered[1].str.contains('SSR')].loc[:, 'ID':].sort_values(by=['ID','Class:Order:Superfamily']).drop_duplicates(subset='ID', keep ='last')

TE_COS_dict = dict(zip(_tmp_subset.loc[:, 'ID'], _tmp_subset.loc[:, 'Class:Order:Superfamily' ]))

_tmp_subset = p_repet_gff_filtered[p_repet_gff_filtered[1].str.contains('SSR')].loc[:, 'ID':].sort_values(by=['ID','Class:Order:Superfamily']).drop_duplicates(subset='ID', keep ='last')

_tmp_dict = dict(zip(_tmp_subset.loc[:, 'ID'], _tmp_subset.loc[:, 'Class:Order:Superfamily' ]))

TE_COS_dict.update(_tmp_dict)
#remove all backslashes from the values as this will conflict with the output later on
for x in TE_COS_dict.keys():
    if '/' in TE_COS_dict[x]:
        value = TE_COS_dict[x]
        print(value)
        TE_COS_dict[x] = value.replace('/','_')
        print(TE_COS_dict[x])

p_repet_gff_filtered.to_csv(out_dir+'/'+genome+'.REPET.long.df', sep='\t', header = None, index=None)

p_repet_gff_filtered['Class:Order:Superfamily'] = p_repet_gff_filtered['ID'].apply(lambda x: TE_COS_dict[x])

print('These are the unique Class:Order:Superfamily classifiers of this dataframe:')
print(p_repet_gff_filtered['Class:Order:Superfamily'].unique())

#have a rough summary of the coverage not considering overlaps.
p_repet_gff_filtered.drop_duplicates(subset=[3,4,'ID'], inplace =True)
p_repet_gff_filtered['Length'] = p_repet_gff_filtered[4] - p_repet_gff_filtered[3]
p_repet_gff_filtered['Class'] = p_repet_gff_filtered.apply(lambda row: TE_classification_filter(row['Class:Order:Superfamily'], 0), axis=1)
p_repet_gff_filtered['Order'] = p_repet_gff_filtered.apply(lambda row: TE_classification_filter(row['Class:Order:Superfamily'], 1), axis=1)
p_repet_gff_filtered['Superfamily'] = p_repet_gff_filtered.apply(lambda row: TE_classification_filter(row['Class:Order:Superfamily'], 2), axis=1)
p_repet_gff_len_COS = p_repet_gff_filtered.groupby(by=['Class','Order','Superfamily'])['Length'].sum()
p_repet_gff_len_S = p_repet_gff_filtered.groupby(by=['Class:Order:Superfamily'])['Length'].sum()

print("This is the summary of overlapping coverage according to Class, Order, Superfamily")
print(p_repet_gff_len_COS)

print("This is the summary of overlapping coverage according to Superfamily")
print(p_repet_gff_len_S)

num_unique_TEs = len(p_repet_gff_filtered[~p_repet_gff_filtered[1].str.contains('SSR')]['ID'].unique())
num_unique_TE_super = len(p_repet_gff_filtered[~p_repet_gff_filtered[1].str.contains('SSR')]['Class:Order:Superfamily'].unique())

print('This is the number of unique TEs: %i\nThis is the number of unique TE superfamilies: %i' % (num_unique_TEs, num_unique_TE_super))

p_repet_gff_filtered.groupby(by=['Class','Order','Superfamily'])['Length'].count() 

p_repet_gff_filtered.groupby(by=['Class:Order:Superfamily'])['Length'].count()

p_repet_gff_filtered.to_csv(out_dir+'/'+genome+'.REPET.long_v2.df', sep='\t', header = None, index=None)

#make new gff files where the ID column is the superfamily level
p_repet_gff_superfamily = p_repet_gff_filtered.iloc[:,:]
p_repet_gff_superfamily[8] = p_repet_gff_superfamily['Class:Order:Superfamily']
p_repet_gff_superfamily.iloc[:,0:9].to_csv(out_dir+'/'+genome+'.REPET.superfamily.gff', sep='\t', header = None, index=None,columns=None)

#make new gff file where the ID column is the TE level
p_repet_gff_TE = p_repet_gff_filtered.iloc[:,:]
p_repet_gff_TE[8] = p_repet_gff_TE['ID']
p_repet_gff_TE.iloc[:,0:9].to_csv(out_dir+'/'+genome+'.REPET.TE.gff', sep='\t', header = None, index=None,columns=None)

#generate the directory structure to safe specific coverage files
os.chdir(out_dir)
TE_types = ['Retrotransposon', 'DNA_transposon', 'noCat', 'SSR']
TE_path = [os.path.join(out_dir, x) for x in TE_types]
TE_path_dict = dict(zip(TE_types, TE_path))
for TE_type in TE_types:
    new_path = os.path.join(out_dir, TE_type)
    if not os.path.exists(new_path):
        os.mkdir(new_path)

# subset the id and safe in specific folder
# return the subsetted file as bedtool
def subset_id(_id, bed_object, repet_prefix):
    #ClassI are retrotransposon form blast
    if 'ClassI:' in _id:
        out_path = TE_path_dict['Retrotransposon']   
    #ClassII are DNA_transponson
    elif 'ClassII' in _id:
        out_path = TE_path_dict['DNA_transposon'] 
    #The rest with '_' should be REPET_TEs
    elif _id == 'noCat':
        out_path = TE_path_dict['noCat']
    #everything without '_' at the end should be SSR
    elif _id == 'SSR':
        out_path = TE_path_dict['SSR']
    out_fn = out_path+'/'+repet_prefix+'.'+_id+'.gff'
    result = bed_object.filter(id_filter, _id).saveas(out_fn)
    cov_fn = out_fn.replace('gff','cov')
    cov = result.genome_coverage(dz=True,g=p_genome_file)
    cov.saveas(cov_fn)
    #_len = len(pd.read_csv(cov_fn, header=None, sep='\t'))
    #_dict[_id] = _len
    #return pybedtools.BedTool(result.fn)

# Next, we create a function to pass only features for a particular
# featuretype.  This is similar to a "grep" operation when applied to every
# feature in a BedTool
def id_filter(feature, _id):
    if feature[8] == _id:
        return True
    return False

repet_prefix_TE = genome+'.REPET.TE'
repet_prefix_S = genome+'.REPET.superfamily'
p_genome_file = genome+'.genome_file'
genome_df = pd.read_csv(p_genome_file, sep='\t', header=None,names=['contig', 'length'])

genome_size = genome_df['length'].sum()

#pull in the classification gff, make classification array, loop over array to save all the cov_dataframes
RE_TE_gff = pybedtools.BedTool(out_dir+'/'+genome+'.REPET.TE.gff')
g_TE = RE_TE_gff.remove_invalid().saveas(out_dir+'/'+genome+'.REPET.TE.bedobject')
#use the blast filtered dataframe as well
RE_S_gff = pybedtools.BedTool(out_dir+'/'+genome+'.REPET.superfamily.gff')
g_S = RE_S_gff.remove_invalid().saveas(out_dir+'/'+genome+'.REPET.superfamily.bedobject')

#use simple loop to loop over the bedcov genome coverage per classification. Keep track if everything is already done.
jobs = []
bed_file = g_S
superfamilies = p_repet_gff_superfamily['Class:Order:Superfamily'].unique()
for superfamily in superfamilies:
    subset_id(superfamily, bed_file, repet_prefix_S)
    print('Doing %s' % superfamily)

cur_dir = os.path.abspath(os.path.curdir)
#this caputures all REPET classifications add the superfamily level
class_cov_files = []
for dirpath, dirname, filenames in os.walk(cur_dir, topdown=True):
    if dirpath == cur_dir:
        continue
    cov_files = [dirpath +'/'+x for x in os.listdir(dirpath) if x.endswith('.cov') and repet_prefix_S in x]
    for file in cov_files:
        class_cov_files.append(file)

#make a large summary dataframe from all the cov files where the last 
df_list =[]
class_cov_files.sort()
for file in class_cov_files:
    print(file)
    tmp_df = pd.read_csv(file, sep='\t', header = None)
    tmp_df["Class:Order:Superfamily"] = file.split('.')[-2]
    tmp_df.drop_duplicates(inplace=True) #drop all the duplicates meaning same position in the genome and same superfamily
    df_list.append(tmp_df)
    print(file.split('.')[-2])

df_REPET_classification = pd.concat(df_list)
df_REPET_classification.to_csv(out_dir+'/'+ repet_prefix_S +'.cov', sep='\t', header =None, index=None)

cov_per_superfamily = df_REPET_classification.pivot_table(values=1, columns= "Class:Order:Superfamily", aggfunc='count')
cov_per_contig_per_superfamily = df_REPET_classification.groupby([0, "Class:Order:Superfamily"])[1].count()

cov_all_TEs = df_REPET_classification.drop_duplicates([0,1]) #this gets ride of the overlap between different TE families and classes
cov_all_TEs = len(cov_all_TEs)


#make superfamily df and add columns for Class, order and superfamily
cov_per_superfamily_df = cov_per_superfamily.append(pd.DataFrame.from_dict({'cov_all_TEs': cov_all_TEs}, orient='index'))
cov_per_superfamily_df.rename(columns={0: 'bp'}, inplace=True)
cov_per_superfamily_df['%'] = cov_per_superfamily_df['bp']/genome_size*100
cov_per_superfamily_df['Class:Order:Superfamily'] = cov_per_superfamily_df.index

cov_per_superfamily_df['Class'] = cov_per_superfamily_df.apply(lambda row: TE_classification_filter(row['Class:Order:Superfamily'], 0), axis=1)
cov_per_superfamily_df['Order'] = cov_per_superfamily_df.apply(lambda row: TE_classification_filter(row['Class:Order:Superfamily'], 1), axis=1)
cov_per_superfamily_df['Superfamily'] = cov_per_superfamily_df.apply(lambda row: TE_classification_filter(row['Class:Order:Superfamily'], 2), axis=1)
cov_per_superfamily_df.to_csv(out_dir+'/'+genome+'.REPET.summary.tab', sep='\t')

#consider combining these cov data frames into classes and orders as well and simply using those as id column, drop duplicats 
#use those as real coverage analysis at those level

cur_dir = os.path.abspath(os.path.curdir)
#this caputures all REPET classifications add the superfamily level
class_cov_files = []
for dirpath, dirname, filenames in os.walk(cur_dir, topdown=True):
    if dirpath == cur_dir:
        continue
    cov_files = [dirpath +'/'+x for x in os.listdir(dirpath) if x.endswith('.cov') and repet_prefix_S in x]
    for file in cov_files:
        class_cov_files.append(file)

#make a large summary dataframe from all the cov files where the last 
df_list =[]
class_cov_files.sort()
for file in class_cov_files:
    tmp_df = pd.read_csv(file, sep='\t', header = None)
    tmp_df["Class"] = file.split('.')[-2].split(':')[0] #parse out the Class from the file name
    tmp_df.drop_duplicates(inplace=True) #drop all the duplicates meaning same position in the genome and same Class
    df_list.append(tmp_df)
    print(file.split('.')[-2].split(':')[0])

df_REPET_classification_class = pd.concat(df_list)
df_REPET_classification_class.drop_duplicates(inplace=True)
df_REPET_classification_class.to_csv(out_dir+'/'+ repet_prefix_S.replace('superfamily', 'Class') +'.cov', sep='\t', header =None, index=None)

cov_per_class = df_REPET_classification_class.pivot_table(values=1, columns= "Class", aggfunc='count')
cov_per_contig_per_class = df_REPET_classification_class.groupby([0, "Class"])[1].count()

#this parsing of of the order is neccessary to eliminate any overlap at the order level. See drop duplicates line 23.
cur_dir = os.path.abspath(os.path.curdir)
#this caputures all REPET classifications add the superfamily level
class_cov_files = []
for dirpath, dirname, filenames in os.walk(cur_dir, topdown=True):
    if dirpath == cur_dir:
        continue
    cov_files = [dirpath +'/'+x for x in os.listdir(dirpath) if x.endswith('.cov') and repet_prefix_S in x]
    for file in cov_files:
        class_cov_files.append(file)

#make a large summary dataframe from all the cov files where the last 
df_list =[]
class_cov_files.sort()
for file in class_cov_files:
    tmp_df = pd.read_csv(file, sep='\t', header = None)
    if ':' in file:
        tmp_df["Order"] = ':'.join(file.split('.')[-2].split(':')[0:2]) #parse out the order from the file name
        print(':'.join(file.split('.')[-2].split(':')[0:2]))
    else:
        tmp_df["Order"] = file.split('.')[-2].split(':')[0]
        print(file.split('.')[-2].split(':')[0])
    tmp_df.drop_duplicates(inplace=True) #drop all the duplicates meaning same position in the genome and same Order
    df_list.append(tmp_df)

df_REPET_orderification_order = pd.concat(df_list)
df_REPET_orderification_order.drop_duplicates(inplace=True)
df_REPET_orderification_order.to_csv(out_dir+'/'+ repet_prefix_S.replace('superfamily', 'Order') +'.cov', sep='\t', header =None, index=None)

cov_per_order = df_REPET_orderification_order.pivot_table(values=1, columns= "Order", aggfunc='count')
cov_per_contig_per_order = df_REPET_orderification_order.groupby([0, "Order"])[1].count()

cov_per_order_df = cov_per_order.append(pd.DataFrame.from_dict({'cov_all_TEs': cov_all_TEs}, orient='index'))

cov_per_order_df.rename(columns={0: 'bp'}, inplace=True)

cov_per_order_df['%'] = round(cov_per_order_df['bp']/genome_size*100, 3)

cov_per_class_df = cov_per_class.append(pd.DataFrame.from_dict({'Total RE coverage': cov_all_TEs}, orient='index'))

cov_per_class_df.rename(columns={0: 'bp'}, inplace=True)

cov_per_class_df['%'] = round(cov_per_class_df['bp']/genome_size*100, 3)

cov_per_class_df.sort_values('%', inplace=True)

plt.style.available

plt.style.use('seaborn-talk')
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(12,15))
fig.suptitle("Repetitive elements in P. striformis f. sp. tritici %s" % genome, fontsize=14, fontweight = 'bold')

#color cycle from color blind people 
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
'#999999', '#e41a1c', '#dede00']

#plot the overall genome coverage by repetitive element category
cov_per_class_df.plot(kind='barh', y='%', ax=ax0,  color='r')
ax0.set_xlim([-5,60])
ax0.legend().set_visible(False)
ax0.set_yticklabels(list(cov_per_class_df.index),fontsize=10, fontweight='bold')
ax0.set_ylabel(ylabel='RE categories', fontsize=14, fontweight='bold')

#plot class I 
classI_df = classI_df = cov_per_superfamily_df[cov_per_superfamily_df['Class'] == 'Retrotransposon'].sort_values('%')

#pick out the colors to do color matching on the order level
tmp_cn = len(classI_df['Order'].unique())
tmp_colors = CB_color_cycle[0:tmp_cn]
tmp_col_dict = dict(zip(classI_df['Order'].unique(), tmp_colors))
classI_df['Color'] = classI_df['Order'].apply(lambda x: tmp_col_dict[x])
classI_df.plot(kind='barh', y = '%', ax=ax1, color=classI_df['Color'])
ax1.set_xlim([-2,25])
ax1.legend().set_visible(False)
ax1.set_yticklabels(list(classI_df.index),fontsize=10, fontweight='bold')
ax1.set_ylabel(ylabel='Class:Order:Superfamily', fontsize=14, fontweight='bold')
ax1.set_title('ClassI: Retrotransposons', fontsize=14, fontweight='bold')

#add tick lables

for p, value in zip(ax1.patches, classI_df['%']):
    ax1.annotate('{0:.3f}'.format(value), (18,p.get_y() * 1.005),fontsize=10, fontweight='bold' )

#plot class II
classII_df = classII_df = cov_per_superfamily_df[cov_per_superfamily_df['Class'] == 'DNA_transposon'].sort_values('%')

#pick out the colors to do color matching on the order level
tmp_cn = len(classII_df['Order'].unique())
tmp_colors = CB_color_cycle[0:tmp_cn]
tmp_col_dict = dict(zip(classII_df['Order'].unique(), tmp_colors))
classII_df['Color'] = classII_df['Order'].apply(lambda x: tmp_col_dict[x])

#plot the class II out
classII_df.plot(kind='barh', y = '%', ax=ax2, color=classII_df['Color'])
ax2.set_xlim([-2,25])
ax2.legend().set_visible(False)
ax2.set_yticklabels(list(classII_df.index),fontsize=10, fontweight='bold')
ax2.set_ylabel(ylabel='Class:Order:Superfamily', fontsize=14, fontweight='bold')
ax2.set_title('ClassII: DNA transposons', fontsize=14, fontweight='bold')
ax2.set_xlabel('% genome coverage', fontsize=14, fontweight='bold')

#add tick lables

for p, value in zip(ax2.patches, classII_df['%']):
    ax2.annotate('{0:.3f}'.format(value), (18 ,p.get_y() * 1.005),fontsize=10, fontweight='bold' )
    
fig.savefig(genome+'.REPET_summary.seaborn-talk.png', dpi=600, bbox_inches="tight")



