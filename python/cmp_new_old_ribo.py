import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

# read in data
len_range = range(22,51)
new_cov_fn = '/data/shangzhong/RibosomeProfiling/ritux/02_cov/tri_cov.txt'
old_cov_fn = '/data/shangzhong/RibosomeProfiling/ritux/previous_data/02_cov/s06_cov.txt'
new_df = pd.read_csv(new_cov_fn,sep='\t',header=0,usecols=[0,5],names=['count','len'])
new_len_count = new_df.groupby('len').sum()
old_df = pd.read_csv(old_cov_fn,sep='\t',header=0,usecols=[0,5],names=['count','len'])
old_len_count = old_df.groupby('len').sum()
# get count at each length
new_c = []
old_c = []
for l in len_range:
    try:
        new_c.append(new_len_count.loc[l,'count'])
    except:
        new_c.append(0)
    try:
        old_c.append(old_len_count.loc[l,'count'])
    except:
        old_c.append(0)
new_total = sum(new_c)
old_total = sum(old_c)
new_cov = [float(p)/new_total for p in new_c]
old_cov = [float(p)/old_total for p in old_c]
# plot data
fig = plt.subplots(figsize=(8,4))
_ = plt.plot(len_range,new_cov,label='new')
_ = plt.plot(len_range,old_cov,label='old')
_ = plt.xticks(len_range)
_ = plt.legend()
_ = plt.title('mapping length distribution')
_ = plt.xlabel('length')
_ = plt.ylabel('portion')

# get count percentage old data
old_cov_fn = '/data/shangzhong/RibosomeProfiling/ritux/previous_data/02_cov/s06_cov.txt'
old_cov_df = pd.read_csv(old_cov_fn,sep='\t',usecols=[0],names=['count'])
old_total = old_cov_df['count'].sum()
old_fn = '/data/shangzhong/RibosomeProfiling/ritux/previous_data/05_cds_utr_count/s06_cov.txt'
old_df = pd.read_csv(old_fn,sep='\t',header=0,index_col=0)
old_count = old_df.sum().div(old_total)
old_count['other'] = 1 - sum(old_count)
# get count percentage for new data
new_cov_fn = '/data/shangzhong/RibosomeProfiling/ritux/02_cov/tri_cov.txt'
new_cov_df = pd.read_csv(new_cov_fn,sep='\t',usecols=[0],names=['count'])
new_total = new_cov_df['count'].sum()
new_fn = '/data/shangzhong/RibosomeProfiling/ritux/05_cds_utr_count/tri_cov.txt'
new_df = pd.read_csv(new_fn,sep='\t',header=0,index_col=0)
new_count = new_df.sum().div(new_total)
new_count['other'] = 1 - sum(new_count)
# plot
df = pd.DataFrame()
df['old'] = old_count
df['new'] = new_count
df.index = ['cds','utr5','utr3','other']

ax = df.plot(kind='bar',title='percentage of reads in each feature')
_ = ax.set_ylabel('portion')

# new_gene_count_fn = '/data/shangzhong/RibosomeProfiling/ritux/06_gene_intron_count/tri_cov.txt'
# new_gene_count_df= pd.read_csv(new_gene_count_fn,sep='\t',header=0,index_col=0)
# new_gene_total = new_gene_count_df.sum().div(new_total)
# new_count = new_count.append(new_gene_total)
# old_gene_count_fn = '/data/shangzhong/RibosomeProfiling/ritux/06_gene_intron_count/tri_cov.txt'
# old_gene_count_df= pd.read_csv(old_gene_count_fn,sep='\t',header=0,index_col=0)
# old_gene_total = old_gene_count_df.sum().div(old_total)
# old_count = old_count.append(old_gene_total)
# old_count

