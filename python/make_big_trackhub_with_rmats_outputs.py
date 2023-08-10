get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from rnaseq import rmats_inclevel_analysis as rmats
from encode import manifest_helpers
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tnrange, tqdm_notebook
import pybedtools

pd.set_option('display.max_columns', 500)

pos_annotations = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.positive.nr.txt')
neg_annotations = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.negative.nr.txt')
all_annotations = glob.glob('/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/*significant.nr.txt')



import pybedtools

def rmats2bedtool_all(annotations):
    """
    Returns merged intervals that contain all regions from all 
    junction counts files. This ensures that regions will be non-overlapping
    and can be used as 'keys' to regions on the epigenome browser.
    """
    intervals = []
    progress = tnrange(len(annotations))
    for pos in annotations:
        nice_name = os.path.basename(pos)
        nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly','')
        nice_name = nice_name.replace('nr.txt','')
        # nice_name = os.path.basename(pos).replace('-SE.MATS.JunctionCountOnly.positive.nr.txt','')
        # nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly.negative.nr.txt','')
        
        df = pd.read_table(pos)
        for col, row in df.iterrows():
            intervals.append(
                pybedtools.create_interval_from_list(
                    [row['chr'], str(row['exonStart_0base']), str(row['exonEnd']), nice_name, str(row['IncLevelDifference']), row['strand']])
            )
        progress.update(1)
    bedtool_all_intervals = pybedtools.BedTool(intervals)
    return bedtool_all_intervals.sort().merge()

def rmats2bedtool(annotation):
    """
    Returns a bedtool object from an rmats annotation
    Uses the exonStart_0base and exonEnd as coordinates.
    """
    intervals = []
    df = pd.read_table(annotation)
    # progress = tnrange(df.shape[0], leave=False)
    for _, row in df.iterrows():
        intervals.append(
            pybedtools.create_interval_from_list(
                [
                    row['chr'], str(row['exonStart_0base']), 
                    str(row['exonEnd']), row['GeneID'], 
                    str(row['IncLevelDifference']), row['strand']
                ]
            )
        )
        # progress.update(1)
    
    return pybedtools.BedTool(intervals)

def transform_individual_rmats_positions(rmats_file, big_merged_bedtool):
    """
    turns individual rmats exon positions into something 
    common to those found in the big merged bedtool file.
    
    Parameters:
    rmats_file : string
        rmats JunctionCountsOnly.txt
    big_merged_bedtool : pybedtools.BedTool
        intervals containing all regions for all annotations being compared.
    
    Returns: dataframe
    """
    
    # just get the "nice name" (RBP_CELL without the extra stuff)
    nice_name = os.path.basename(rmats_file)
    
    # nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly..nr.txt','')
    # nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly.negative.nr.txt','')
    nice_name = nice_name.split('-')
    nice_name = '{}_{}'.format(nice_name[0], nice_name[2])
    
    # for each rmats file, intersect with the merged bedtool to bin regions into those that are common amongst all
    individual_rmats_bedtool = rmats2bedtool(rmats_file).sort()
    intersected = individual_rmats_bedtool.intersect(big_merged_bedtool, wb=True).to_dataframe()
    
    # thickStart, thickEnd, itemRgb actually contain the 'key' common regions from the big_merged_bedtool intersection.
    intersected['chrom'] = intersected['thickStart']
    intersected['start'] = intersected['thickEnd']
    intersected['end'] = intersected['itemRgb']
    
    # re-format so that it's a proper dataframe, and re-name the 'score' column to be that of the name of the RBP. 
    intersected = intersected[['chrom','start','end','name','score','strand']]
    intersected.columns = ['chrom','start','end','name','{}'.format(nice_name),'strand']
    intersected.set_index(['chrom','start','end','name','strand'], inplace=True)
    return intersected

def merge_all_rmats_transformed(all_annotations, big_merged_bedtool):
    """
    merge all dpsi for common regions (as described in big_merged_bedtool) into one dataframe. 
    """
    progress = tnrange(len(all_annotations))
    
    # do this once to easily/automatically populate the index.
    merged = transform_individual_rmats_positions(all_annotations[0], big_merged_bedtool)
    progress.update(1)
    
    # foreach subsequent file, merge (outer join to not miss any) files into merged.
    for annotation in all_annotations[1:]:
        df = transform_individual_rmats_positions(annotation, big_merged_bedtool)
        merged = pd.merge(merged, df, how='outer', left_index=True, right_index=True)
        progress.update(1)
    return merged

# get the full list of conjoined exon intervals
big_merged_bedtool = rmats2bedtool_all(all_annotations)

merged = merge_all_rmats_transformed(all_annotations, big_merged_bedtool)

merged.to_csv('/home/bay001/projects/encode/analysis/rnaseq_trackhub_attempt2/merged_from_rmats_nonredundant.txt', sep='\t')

merged.loc['chr1', 1191424, 1191561, 'ENSG00000160087.16', '-'].dropna()

merged.head()

from collections import defaultdict

def format_df_to_trackhub(df, qcat_dict, out_file):
    """
    kind of a messy way to re-format the dataframe
    """
    with open(out_file, 'w') as o:
        count = 1
        progress = tnrange(df.shape[0])
        for col, row in df.iterrows():
            row_str = "{}\t{}\t{}\t".format(row['chrom'], row['start'], row['end'])
            row_str = row_str + 'id:{},qcat:'.format(count)
            qcat_str = ""
            row.sort_values(inplace=True)
            for i in row.index:
                if i in qcat_dict.keys() and not pd.isnull(row[i]): # there is a value
                    
                    qcat_str = qcat_str + '[{},{}], '.format(row[i], qcat_dict[i][0])

            qcat_str = qcat_str[:-2]
            if qcat_str != '':
                o.write(row_str + '[ {} ]\n'.format(qcat_str))
            count += 1
            progress.update(1)
    return 0

def rbp_to_qcat(json_like):
    """
    turns this json like file into a dictionary
    with rbp names as keys and category ID, color as values
    """
    categories = defaultdict(list)
    with open(json_like, 'r') as f:
        for line in f:
            if line.startswith('\t'):
                try:
                    line = line.replace('\'','')
                    category, rbp = line.replace('[','').replace(']','').split(':')
                    rbpname, rbpcolor, _ = rbp.split(',')
                    categories[rbpname] = [int(category.replace('\t','')), rbpcolor]
                except ValueError:
                    print(line)
    return categories

def return_json_id_from_merged_column(column):
    """
    only difference between this and jxc function in the junctioncountsonly notebook is the - and _
    """
    rbp_name, rbp_cell = column.split('_')
    return "{}_{}_01".format(rbp_name, rbp_cell) # we don't care about replicates; rmats is one file per 2 reps



def merged_column_to_qcat_elements(column, qcat_dict):
    current = len(qcat_dict.keys())
    # print(return_json_id_from_merged_column(column))
    values = qcat_dict[
        return_json_id_from_merged_column(column)
    ]
    if values != []:
        return values #[-1, "#0000FF"] # values
    else:
        return [-1, "#0000FF"]
    
json_file = '/home/bay001/projects/encode/analysis/rnaseq_trackhub/combined_10bpfull.datahub.pos'
qcat_dict = rbp_to_qcat(json_file)



# this ensures that a unique identifier will be assigned to any shRNA rnaseq expt not already assigned in clip data.
from collections import defaultdict

colors = sns.color_palette("husl", len(qcat_dict)).as_hex()

new_qcat_dict = defaultdict(list)
counter = len(qcat_dict)
print('total IDs already assigned: {}'.format(counter))
for column in merged.columns:
    if 'HepG2' in column or 'K562' in column:
        exists, qcat_id_color = merged_column_to_qcat_elements(column, qcat_dict)
        if exists == -1:
            counter += 1
            new_qcat_dict[column] = counter, colors[counter-len(qcat_dict)].upper()
        else:
            new_qcat_dict[column] = exists, qcat_id_color

qcat_df = pd.DataFrame(new_qcat_dict).T.reset_index().sort_values(by=0)
qcat_df.head()

out_file = '/home/bay001/projects/encode/analysis/rnaseq_trackhub_attempt2/trackhub_merged_from_rmats_nonredundant.jsonlike'
format_df_to_trackhub(merged.reset_index(), new_qcat_dict, out_file)

datahub_file = '/home/bay001/projects/encode/analysis/rnaseq_trackhub_attempt2/trackhub_merged_from_rmats_nonredundant.datahub.txt'
with open(datahub_file, 'w') as f:
    f.write('[\n')
    f.write('{\n')
    f.write('type:\'quantitativeCategorySeries\',\n')
    f.write('name:\'test_hub_please_ignore\',\n')
    f.write('height:500,\n')
    f.write('url:\"https://s3-us-west-1.amazonaws.com/washington-university-epigenome-browser-trackhub-test-2/trackhub_merged_from_rmats_nonredundant.sorted.jsonlike.gz\",\n')
    f.write('backgroundcolor:\'#FFFFFF\',\n')
    f.write('mode:\'show\',\n')
    f.write('categories:{\n')
    ### write the actual stuff
    for _, row in qcat_df.iterrows():
        f.write('\t\'{}\':[\'{}\',\'{}\'],\n'.format(
            row[0], row['index'], row[1]
            ))
    f.write('\t},\n')
    f.write('},\n')
    f.write(']')
    

sorted_out_file = '/home/bay001/projects/encode/analysis/rnaseq_trackhub_attempt2/trackhub_merged_from_rmats_nonredundant.sorted.jsonlike'
get_ipython().system(' sort -k1,1 -k2,2n $out_file > $sorted_out_file')

get_ipython().system(' bgzip -f $sorted_out_file')

gz = '{}.gz'.format(sorted_out_file)
get_ipython().system(' tabix -p bed $gz')



