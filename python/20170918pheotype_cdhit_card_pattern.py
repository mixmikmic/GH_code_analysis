# import 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# set plotting style
mpl.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# loading needed dataframe
df = pd.read_pickle("../../cdhitResult/ec0913_df")
cluster_detail = pd.read_pickle("../../cdhitResult/cluster_detail_tmp1010")
ris = pd.read_pickle("../../../data/anno_sps_df")
card = pd.read_pickle("../../cdhitResult/ec0913_card_df")



# join RIS and cdhit absence presence pattern
ris_need = ris[["Genome ID", "Resistant Phenotype", "Antibiotic"]]
df_ris = pd.merge(df, ris_need, left_index = True, right_on = "Genome ID")
df_ris

# checking data abundance
count = df_ris[["Cluster 0", "Antibiotic"]].groupby(by = "Antibiotic").count()
abx_abundance = count.sort_values(by = "Cluster 0")
abx_abundance

# card absence presence for cdhit
# card_detail containing every card gene and its cluster number, cog number
card_detail = pd.merge(card, cluster_detail, right_on = "representing gene header", left_index = True, how = 'outer')
card_yn = pd.DataFrame()
card_yn = card_yn.append(card_detail['ARO'].notnull())

blast = pd.read_pickle("/home/hermuba/resistanceExp/data/blastp_gi_1022")

# calculate p value
from scipy import stats
def p_value(abx):
    abx_choose = df_ris.loc[df_ris['Antibiotic'] == abx]
    
    # seperate into r and not-r
    r = abx_choose.loc[abx_choose['Resistant Phenotype'] == "Resistant"]
    not_r = abx_choose.loc[abx_choose['Resistant Phenotype'] != "Resistant"]
    
    d = pd.DataFrame(columns = ['t_stat', 'p_value'])
    # loop around cluster
    for cluster_name in abx_choose.columns[0:15949]:
        d.loc[cluster_name, 't_stat'] = stats.ttest_ind(r[cluster_name], not_r[cluster_name])[0]
        d.loc[cluster_name, 'p_value'] = stats.ttest_ind(r[cluster_name], not_r[cluster_name])[1]
    
    # return
    return(d)

def identify_p(d):
    d = d.loc[d['p_value'] < 0.05]
    d = pd.merge(d, cluster_detail, right_index = True, left_index = True, how = 'inner')
    d = pd.merge(d, blast, right_index = True, left_on = 'representing gene header', how = 'inner')
    return(d)

# count cog (borrow from 1010_just_pangenome)
def cog_dis(df): # parse cog
    result = pd.DataFrame(np.zeros((1,26)), columns = ['J','A','K','L','B','D','Y','V','T','M','N','Z','W','U','O','C','G','E','F','H','I','P','Q','R','S','NaN'])
    cog = list(df.loc[df['cog'].notnull()]['cog'])
    for str in cog:
        for s in str:
            result[s] += 1
    result['NaN'] = df.shape[0]-len(cog)
    return(result)

drug = ['cefepime', 'ceftazidime', 'ampicillin/sulbactam', 'cefazolin', 'ampicillin', 'trimethoprim/sulfamethoxazole','ciprofloxacin', 'gentamicin', 'meropenem']
signi_char = pd.DataFrame(index = drug, columns = ['significant gene', 'CARD', 'mean prevalance', 'std prevalance', 'hypothetical'])
signi_cog = pd.DataFrame(columns = ['J','A','K','L','B','D','Y','V','T','M','N','Z','W','U','O','C','G','E','F','H','I','P','Q','R','S','NaN'],
                        index = drug)
signi_prev = {}
loc = 0
for i in drug:
    p_selected = identify_p(p_value(i))
    # save the dataframe
    p_selected.to_excel("/home/hermuba/resistanceExp/EcoliGenomes/figures/" + i.replace('/','') + "_significant.xlsx")
    
    # calculate some properties
    how_many = p_selected.shape[0]
    signi_char.loc[i, 'significant gene'] = how_many
    signi_char.loc[i, 'CARD'] = p_selected.loc[p_selected['card_portion']>0].shape[0]
    signi_char.loc[i, 'mean prevalance'] = p_selected['prevalance'].mean()
    signi_char.loc[i, 'std prevalance'] = p_selected['prevalance'].mean()
    signi_char.loc[i, 'hypothetical'] = p_selected.loc[p_selected['title'].str.contains("hypothetical")].shape[0]/how_many
    
    # calculate cog distribution
    signi_cog.iloc[loc, :] = cog_dis(p_selected).values # concat
    loc = loc + 1

signi_char.to_excel("/home/hermuba/resistanceExp/EcoliGenomes/figures/significant_property.xlsx")

signi_cog.to_excel("/home/hermuba/resistanceExp/EcoliGenomes/figures/significant_cog.xlsx")

# calculate rs ratio; input antibiotic name; output: r/s ratio, index sorted by r/s ratio
def rs_ratio(abx):
    abx_choose = df_ris.loc[df_ris['Antibiotic'] == abx]
    
    freq = abx_choose.groupby(by = "Resistant Phenotype").mean() #frequency == 1: core genome
    freq_only_cluster = freq.iloc[:, 0:15949]
    freq_only_cluster = freq_only_cluster.loc[freq_only_cluster['Cluster 1'].notnull()]
    
    ratio = freq_only_cluster.loc['Resistant', :]/freq_only_cluster.loc['Susceptible', :]
    # this is the r/s ratio
    
    i = ratio.sort_values().index
    # index sort by r/s ratio

    return(i, ratio)
    
def plot_colormap(abx,):
        
    sort = abx_choose.sort_values("Resistant Phenotype") # sort by Resistant Phenotype
    only_cluster = sort.iloc[:,0:15949] # remove "Resistant Phenotype", leave only 0101 for cdhit pattern
    
    
    
    plt.figure(figsize = (40,6))
    #absense presence pattern
    plt.subplot(4, 1, 1)
    sort_by_ratio = only_cluster[i]
    plt.pcolor(sort_by_ratio)
    plt.yticks(np.arange(0.5, len(sort.index), 1), sort["Resistant Phenotype"])
    plt.xticks(np.arange(0.5, len(sort_by_ratio.columns), 1), sort_by_ratio.columns)
    plt.title(abx + ' absence presence pattern')
    
    #frequency map 
    plt.subplot(4, 1, 2)
    sort_by_ratio = freq_only_cluster[i]
    plt.pcolor(sort_by_ratio)
    plt.yticks(np.arange(0.5, len(freq_only_cluster.index), 1), freq_only_cluster.index)
    plt.xticks(np.arange(0.5, len(sort_by_ratio.columns), 1), sort_by_ratio.columns)
    plt.title(abx + ' frequency map') 
    
    # card yes or no
    plt.subplot(4,1,3)
    sort_by_ratio = card_yn[i]
    plt.pcolor(sort_by_ratio)
    plt.ylabel('CARD annotation')
    
    # card portion
    plt.subplot(4,1,4)
    sort_by_ratio = card_yn[i]
    plt.pcolor(sort_by_ratio)
    plt.ylabel('CARD annotation')
    
    plt.show()

    

stra_by_abx("meropenem")

stra_by_abx("imipenem")

stra_by_abx("gentamicin")

stra_by_abx("ciprofloxacin")

stra_by_abx("cefepime")

stra_by_abx("cefazolin")

def rs_card_ratio(abx):
    abx_choose = df_ris.loc[df_ris['Antibiotic'] == abx]
    
    freq = abx_choose.groupby(by = "Resistant Phenotype").mean() #frequency == 1: core genome
    freq_only_cluster = freq.iloc[:, 0:15949]
    freq_only_cluster = freq_only_cluster.loc[freq_only_cluster['Cluster 1'].notnull()]
    
    ratio = freq_only_cluster.loc['Resistant', :]/freq_only_cluster.loc['Susceptible', :]
    i = ratio.sort_values().index

    
    sort = abx_choose.sort_values("Resistant Phenotype")
    only_cluster = sort.iloc[:,0:15949]
    
    
    
    plt.figure(figsize = (40,2))
    plt.title(abx + ' CARD pattern')
    # card yes or no
    plt.subplot(2,1,1)
    yn = card_yn[i]
    plt.pcolor(yn, cmap = "magma")
    plt.ylabel('CARD abs')
    
    # card portion
    plt.subplot(2,1,2)
    sr = pd.DataFrame()
    sr = sr.append(cluster_detail['card_portion'])
    sr = sr[i]
    plt.pcolor(sr, cmap = "magma")
    plt.ylabel('CARD portion')
    
    
    plt.show()
    return(yn, sr)



