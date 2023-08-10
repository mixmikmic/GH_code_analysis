get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
import pandas as pd
import os
import glob
from tqdm import tnrange, tqdm_notebook

wd = '/home/bay001/projects/encode/analysis/overlapping_peaks_files/use_max_reiterate'
files = glob.glob(os.path.join(wd,'*.csv'))
print("number of files: ",len(files))

master_table = pd.read_table(
    '/home/bay001/projects/encode/analysis/overlapping_peaks_files/master_list/ENCODE_CLIPperv2_20161120_peaksALLvsALLoverlap.csv',
    index_col=0
)


img_dir = '/home/bay001/projects/encode/analysis/overlapping_peaks_files/use_max_reiterate'

def get_max(df, s1, s2):
    """
    Gets max of A overlapping % B or B overlapping A
    """
    return max(df.loc[s1,s2],df.loc[s2,s1])

def make_individual_files(wd, master_table):
    """
    Creates individual files for each rbp A 
    Containing the max overlap of either:
    A peaks overlapping B peaks, or B peaks overlapping A peaks.
    
    Params:
    Master list from eric containing all overlaps for peaks in all expts.
    
    Returns: 
    Tabbed file
    """
    progress = tnrange(len(master_table.columns))
    for col in master_table.columns: # for each column
        samples = OrderedDict()
        sample = OrderedDict()
        for idx in master_table.loc[col].index: # for each index
            sample[idx] = get_max(master_table, col, idx)
        samples[col] = sample
        df = pd.DataFrame(samples)
        df.to_csv(os.path.join(wd,'{}.csv'.format(col)), sep='\t')
        progress.update(1)
        
def make_individual_files_without_taking_max(wd, master_table):
    """
    Creates individual files for each rbp A 
    Containing the overlap of rbp A peaks to all other rbp peaks
    
    Params:
    Master list from eric containing all overlaps for peaks in all expts.
    
    Returns: 
    Tabbed file 
    """
    for col in master_table.columns: # for each column
        df = pd.DataFrame(master_table[col])
        df.to_csv(os.path.join(wd,'{}.csv'.format(col)), sep='\t')

make_individual_files(wd, master_table)

def rep(string):
    """ expect filename like: 439_02_KHSRP_K562_02.csv """
    return string.split('_')[-1]

def rbp(string):
    """ expect filename like: 439_02_KHSRP_K562_02.csv """
    parts = string.split('_')
    parts = '{}_{}'.format(parts[2],parts[3])
    return parts

def get_rep(row):
    return rep(row['intersecting_rbp'])

def get_rbp(row): 
    return rbp(row['intersecting_rbp'])

def add_rep_and_rbp_info(df):
    df.columns = ['intersecting_rbp','Rep']
    df['rep'] = df.apply(get_rep,axis=1)
    df['rbp'] = df.apply(get_rbp,axis=1)
    return df

def split_and_groupby_rep(df):
    rep1 = df.groupby(['rep']).get_group('01')
    rep2 = df.groupby(['rep']).get_group('02')
    merged = pd.merge(rep1,rep2,how='left', on='rbp')
    merged.set_index('rbp',inplace=True)
    return merged

def purge_same_rep(df,col):
    """
    Remove the rbp values overlapping with itself 
    (will always be 1 regardless)
    """
    df.loc[df['Unnamed: 0']==col,col] = 0
    return df

def get_avg(row):
    """
    Returns the average between the two reps
    """
    return (row['Rep_x'] + row['Rep_y'])/2.0

def iterate_over_files_and_make_cross_corr_barh_draft1(files):
    """
    Iterates over all files created from either:
        make_individual_files()
        make_individual_files_without_taking_max()
    
    plots the barchart, may be deprecated
    """
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        table = pd.read_table(f)
        df = pd.DataFrame(table)
        return df
        # df = purge_same_rep(df, df.columns[1])
        df = add_rep_and_rbp_info(df)
        df = split_and_groupby_rep(df)
        df['avg'] = df.apply(get_avg,axis=1)
        df.rename(columns={'Rep_x':"Rep 1", "Rep_y":"Rep 2"},inplace=True)
        same_rep_values = df.ix[(rbp(name))]
        df.sort_values(by=['avg'],inplace=True,ascending=False)
        df.drop(rbp(name),inplace=True)
        df.head()
        df2 = pd.DataFrame(same_rep_values).T

        dfx = pd.concat([df2,df])
        # same_rep_values"""
        # df2 = pd.DataFrame(same_rep_values).T
        # df2

        if int(rep(name)) == 1:
            dfx.loc[rbp(name),'Rep 1'] = 0
        else:
            dfx.loc[rbp(name),'Rep 2'] = 0

        dfy = dfx[['Rep 1','Rep 2']].head(25).iloc[::-1]
        dfy.plot(kind="barh", figsize=(10,5), rot=0)
        plt.ylabel('Top 25 concordantly bound RBPs')
        plt.xlabel('Fraction of overlapping peaks')
        # plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir,'{}.png'.format(name)))

files = glob.glob(os.path.join(wd, '*.csv'))
rbps = []  # the name of all rbps (no reps). Should be 362/2 = 181 total
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    rbps.append(rbp(name))
rbps = list(set(rbps))
print("number of rbps: {}".format(len(rbps)))

def concat_reps(df):
    """
    reformat the dataframe so that for one replicate, 
    """
    # df = purge_same_rep(df, df.columns[1])
    df = add_rep_and_rbp_info(df)
    df = split_and_groupby_rep(df)
    df['avg'] = df.apply(get_avg,axis=1)
    df.rename(columns={'Rep_x':"Rep 1", "Rep_y":"Rep 2"},inplace=True)
    df.sort_values(by=['avg'],inplace=True,ascending=False)
    """
    same_rep_values = df.ix[(rbp(name))]
    df.drop(rbp(name),inplace=True)
    df2 = pd.DataFrame(same_rep_values).T

    dfx = pd.concat([df2,df])
    if int(rep(name)) == 1:
        dfx.loc[rbp(name),'Rep 1'] = 0
    else:
        dfx.loc[rbp(name),'Rep 2'] = 0

    return dfx"""
    return df

def join_reps(r1, r2):
    """
    Returns a merged table with left-merge = rep1, right-merge = rep2
    Merges on index names (rbp name)
    """
    name1 = os.path.splitext(os.path.basename(r1))[0]
    name2 = os.path.splitext(os.path.basename(r2))[0]
    df1 = pd.read_table(r1)
    df2 = pd.read_table(r2)
    df1 = concat_reps(df1)
    df2 = concat_reps(df2)
    df1.columns = ["Rep 1"+'-'+col for col in df1.columns]
    df2.columns = ["Rep 2"+'-'+col for col in df2.columns]
    merged = pd.merge(df1, df2, how='left', left_index=True, right_index=True)
    return merged

def get_all_avg(row):
    """
    Returns the average of all relevant rows.
    """
    sum_of_relevant_rows = row['Rep 1-Rep 1']+row['Rep 1-Rep 2']+row['Rep 2-Rep 1']+row['Rep 2-Rep 2']
    return sum_of_relevant_rows/4.0

def plot_barchart(wd, img_dir, rbps):
    """
    Iterate foreach rbp in the folder, plot the barchart
    """
    progress = tnrange(len(rbps))
    for rbp_name in rbps:

        f, ax = plt.subplots()
        # ax.set_xlabel('X LABEL')    

        cols = sns.color_palette("hls", 8)
        # use the rbp name to glob both replicates
        repfiles = glob.glob(os.path.join(wd,'*{}*.csv'.format(rbp_name)))  # get both reps
        repfiles = (sorted(repfiles)) # so the '_01' is always before '_02'
        assert len(repfiles) == 2
        r1 = repfiles[0]
        r2 = repfiles[1]
        r_all = join_reps(r1,r2) # join two reps
        r_all['avg'] = r_all.apply(get_all_avg,axis=1) # get avg
        r_all.sort_values(by=['avg'],inplace=True,ascending=False) # sort by highest average
        r_all_head = r_all.head(25) # get the top 25
        r_all_head = r_all_head[['Rep 1-Rep 1','Rep 1-Rep 2','Rep 2-Rep 1','Rep 2-Rep 2']].iloc[::-1] # subset just these columns
        r_all_head.columns = [rbp_name + " " + c for c in r_all_head.columns] # rename '1' in 1vs2 columns
        r_all_head.columns = [c.replace('-',' - (B) ') for c in r_all_head.columns] # rename the '2' in 1vs2 columns
        r_all_head.index.name = r_all_head.index.name + " (B)" # rename the label to be more clear
        r_all_head['{} Rep 1 - (B) Rep 1'.format(rbp_name)].ix[rbp_name] = 0
        r_all_head['{} Rep 2 - (B) Rep 2'.format(rbp_name)].ix[rbp_name] = 0
        r_all_head['{} Rep 2 - (B) Rep 1'.format(rbp_name)].ix[rbp_name] = 0
        r_all_head.plot(
            kind="barh", 
            figsize=(10,15), 
            rot=0, 
            color=[cols[0],cols[1],cols[4],cols[5]],
            fontsize=12,
            legend=False,
            ax=ax
        ) # color as in the above specs
        ax.set_xlim(0,1)
        vals = ax.get_xticks()
        ax.set_xlabel('Fraction overlap')    
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xticklabels(['{:3.2f}%'.format(x*100) for x in vals])
        plt.legend(fontsize=14,loc=4)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir,'{}_RBPconcordancy.png'.format(rbp_name)))
        progress.update(1)

plot_barchart(wd, img_dir, rbps)







