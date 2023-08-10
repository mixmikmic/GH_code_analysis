get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import palettable as pal
import seaborn as sns

sra_run = pd.read_csv('../SraRunInfo_719.csv')
sra_map = sra_run[['Run', 'SampleName']]

# reference-based transrate evaluation
file_dib_v_ncgr = "../assembly_evaluation_data/ncgr_combined_transrate_reference_all.csv"
file_ncgr_v_dib = "../assembly_evaluation_data/ncgr_combined_transrate_reverse_all.csv"

           

# Load in df and add the mmetsp/sra information
dib_v_ncgr = pd.read_csv(file_dib_v_ncgr,index_col="Run")
#dib_v_ncgr = pd.merge(sra_map, dib_v_ncgr, on='Run')

ncgr_v_dib = pd.read_csv(file_ncgr_v_dib,index_col="Run")
#ncgr_v_dib = pd.merge(sra_map, ncgr_v_dib, on='SampleName')

ncgr_v_dib.head()

dib_v_ncgr.head()

#Set indexing value
#ncgr_v_dib = ncgr_v_dib.set_index('Run')
#dib_v_ncgr = dib_v_ncgr.set_index('Run')

#dib_busco = dib_busco.set_index('Run')
#ncgr_busco = ncgr_busco.set_index('Run')

#dib_transrate = dib_transrate.set_index('Run')
#ncgr_transrate = ncgr_transrate.set_index('Run')

dib_v_ncgr = dib_v_ncgr.drop_duplicates()
ncgr_v_dib = ncgr_v_dib.drop_duplicates()

def scatter_diff(df1, df2, column, fig, ax, df1name = 'df1', df2name = 'df2', 
                 color1='#566573', color2='#F5B041', ymin=0, ymax=1, ypos=.95):
    # plot scatter differences between two dfs with the same columns
    # create new df for data comparison
    newdf = pd.DataFrame()
    newdf[df1name] = df1[column]
    newdf[df2name] = df2[column]
    newdf = newdf.dropna()
    newdf = newdf.drop_duplicates()
    # plot with different colors if df1 > or < then df2
    newdf.loc[newdf[df1name] > newdf[df2name], [df1name, df2name]].T.plot(ax=ax, legend = False, 
                                                                          color = color1, lw=2)
    newdf.loc[newdf[df1name] <= newdf[df2name], [df1name, df2name]].T.plot(ax=ax, legend = False, 
                                                                           color = color2, alpha = 0.5, lw=2)
    ax.text(-.1, ypos, str(len(newdf.loc[newdf[df1name] > newdf[df2name]])), 
            color= color1, fontsize='x-large', fontweight='heavy')
    ax.text(.95, ypos, str(len(newdf.loc[newdf[df1name] <= newdf[df2name]])), 
            color= color2, fontsize='x-large', fontweight='heavy')

    # aesthetics 
    ax.set_xlim(-.15, 1.15)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([0,1])
    ax.set_xticklabels([df1name, df2name], fontsize='large', fontweight='bold')
#     ax.set_ylabel(column, fontsize='x-large')
    return newdf, fig, ax
    

def violin_split(df, col1, col2, fig, ax, color2='#566573', color1='#F5B041', ymin=0, ymax=1):
    #create split violine plots
    v1 = ax.violinplot(df[col1],
                   showmeans=False, showextrema=False, showmedians=False)
    for b in v1['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color(color2)
        b.set_alpha(0.85)
    v2 = ax.violinplot(df[col2],
                   showmeans=False, showextrema=False, showmedians=False)
    for b in v2['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(color1)
        b.set_alpha(0.85)
    ax.set_xticks([])
    ax.set_ylim([ymin, ymax])
    
def create_plots(df1, df2, column, col_title, df1name = 'NCGR', df2name = 'DIB', ymax = 1, ymin = 0, ypos = 0.95):
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(6,6)
    outdf, fig, ax = scatter_diff(df1, df2, column = column, 
                 fig = fig, ax = axs[0], df1name = df1name, df2name = df2name, 
                                  ymax = ymax, ymin = ymin, ypos = ypos)
    violin_split(outdf, df1name, df2name, fig, axs[1], ymin = ymin, ymax = ymax)
    fig.suptitle(col_title, fontsize = 'x-large', fontweight = 'bold')
    return outdf, fig, ax

p_refs_with_CRBB, fig, ax = create_plots(ncgr_v_dib, dib_v_ncgr, 'p_refs_with_CRBB', 'Proportion of contigs with CRB-BLAST')

p_refs_with_CRBB.loc[p_refs_with_CRBB.DIB > p_refs_with_CRBB.NCGR]

# Suspecting that dib assemblies have larger numbers of smaller contigs, 
# and ncgr assemblies have smaller numbers of larger contigs
# this looks at the # of total contigs assembled
# looking at 72 for which the ncgr (as reference) had higher p_refs_with_CRBB than dib (as reference)
SRR_id = "SRR1294417"
col = "n_seqs"
print(ncgr_v_dib.loc[SRR_id,col])
print(dib_v_ncgr.loc[SRR_id,col])
# looks like there is a discrepancy in the number of contigs assembled by ncgr (higher)

reference_coverage, fig, ax = create_plots(ncgr_v_dib, dib_v_ncgr, 'reference_coverage', 'Reference coverage'
                                           , ymax = 0.6, ypos = 0.55)

linguistic_complexity, fig, ax = create_plots(ncgr_v_dib, dib_v_ncgr, 'linguistic_complexity', 'Linguistic complexity', ymax=0.35, ypos=0.025)

n50, fig, ax = create_plots(ncgr_v_dib, dib_v_ncgr, 'n50', 'n50', ymax=4000)

mean_orf_percent, fig, ax = create_plots(ncgr_v_dib, dib_v_ncgr, 'mean_orf_percent', 'Mean ORF percent',ymax=100, ypos=0.5)

n_seqs, fig, ax = create_plots(ncgr_v_dib, dib_v_ncgr, 'n_seqs', 'Number of contigs',ymax=200000, ypos=0.5)

n_seqs.loc[n_seqs.DIB < 5000]

busco_scores, fig, ax = create_plots(ncgr_busco,dib_busco,'Complete_BUSCO_perc', 'BUSCO percentages',ymax=1.0, ypos=0.8)

busco_scores.loc[busco_scores.DIB < 0.1]

score,fig, ax = create_plots(ncgr_transrate,dib_transrate, 'score', 'Transrate scores',ymax=.6, ypos=0.55)

score.loc[score.DIB < 0.1]



