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
file_trinity2016_v_trinity2014 = "../assembly_evaluation_data/trinity2014_trinity2.2.0_transrate_reference.csv"
file_trinity2014_v_trinity2016 = "../assembly_evaluation_data/trinity2014_trinity2.2.0_transrate_reverse.csv"

# Load in df and add the mmetsp/sra information
trinity2016_v_trinity2014 = pd.read_csv(file_trinity2016_v_trinity2014,index_col="Run")
trinity2014_v_trinity2016 = pd.read_csv(file_trinity2014_v_trinity2016,index_col="Run")

trinity2016_v_trinity2014.head()

trinity2014_v_trinity2016.head()

trinity2016_v_trinity2014 = trinity2016_v_trinity2014.drop_duplicates()
trinity2014_v_trinity2016 = trinity2014_v_trinity2016.drop_duplicates()

def scatter_diff(df1, df2, column, fig, ax, df1name = 'df1', df2name = 'df2', 
                 color1='#566573', color2='#F5B041', ymin=0, ymax=1, ypos=.95):
    # plot scatter differences between two dfs with the same columns
    # create new df for data comparison
    newdf = pd.DataFrame()
    newdf[df1name] = df1[column]
    newdf[df2name] = df2[column]
    newdf = newdf.dropna()
    newdf = newdf.drop_duplicates()
    # plot with different colors if df1 > or < than df2
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
    
def create_plots(df1, df2, column, col_title, df1name = 'trinity2014', df2name = 'trinity2016', ymax = 1, ymin = 0, ypos = 0.95):
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(6,6)
    outdf, fig, ax = scatter_diff(df1, df2, column = column, 
                 fig = fig, ax = axs[0], df1name = df1name, df2name = df2name, 
                                  ymax = ymax, ymin = ymin, ypos = ypos)
    violin_split(outdf, df1name, df2name, fig, axs[1], ymin = ymin, ymax = ymax)
    fig.suptitle(col_title, fontsize = 'x-large', fontweight = 'bold')
    return outdf, fig, ax

p_refs_with_CRBB, fig, ax = create_plots(trinity2014_v_trinity2016,trinity2016_v_trinity2014, 'p_refs_with_CRBB', 'Proportion of contigs with CRB-BLAST')

p_refs_with_CRBB.loc[p_refs_with_CRBB.trinity2016 < p_refs_with_CRBB.trinity2014]

reference_coverage, fig, ax = create_plots(trinity2014_v_trinity2016,trinity2016_v_trinity2014, 'reference_coverage', 'Reference coverage'
                                           , ymax = 0.9, ypos = 0.9)



linguistic_complexity, fig, ax = create_plots(trinity2014_v_trinity2016,trinity2016_v_trinity2014,  'linguistic_complexity', 'Linguistic complexity', ymax=0.25, ypos=0.025)

mean_orf_percent, fig, ax = create_plots(trinity2014_v_trinity2016,trinity2016_v_trinity2014,  'mean_orf_percent', 'Mean ORF percent',ymax=100, ypos=0.5)

n_seqs, fig, ax = create_plots(trinity2014_v_trinity2016,trinity2016_v_trinity2014, 'n_seqs', 'Number of contigs',ymax=60000, ypos=55000)

