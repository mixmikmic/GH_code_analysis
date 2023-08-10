get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import palettable as pal
import seaborn as sns

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

sra_run = pd.read_csv('../SraRunInfo_719.csv')
sra_map = sra_run[['Run', 'SampleName']]

# reference-based transrate evaluation
file_combined_v_SRA = "../assembly_evaluation_data/combined_transrate_reference.csv"
file_SRA_v_combined = "../assembly_evaluation_data/combined_transrate_reverse.csv"
file_combined_v_ncgr = "../assembly_evaluation_data/ncgr_combined_transrate_reference.csv"
file_ncgr_v_combined = "../assembly_evaluation_data/ncgr_combined_transrate_reverse.csv"

# Load in df and add the mmetsp/sra information
combined_v_SRA = pd.read_csv(file_combined_v_SRA,index_col="Run")
SRA_v_combined = pd.read_csv(file_SRA_v_combined,index_col="Run")
combined_v_ncgr = pd.read_csv(file_combined_v_ncgr,index_col="Run")
ncgr_v_combined = pd.read_csv(file_ncgr_v_combined,index_col="Run")

SRA_v_combined = SRA_v_combined.drop_duplicates()
combined_v_SRA = combined_v_SRA.drop_duplicates()
combined_v_ncgr = combined_v_ncgr.drop_duplicates()
ncgr_v_combined = ncgr_v_combined.drop_duplicates()

def scatter_diff(df1, df2, column, fig, ax, df1name = 'df1', df2name = 'df2', 
                 color1='#566573', color2='#2ecc71', ymin=0, ymax=1, ypos=.95):
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
    #create split violin plots
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
    
def create_plots(df1, df2, column, col_title, df1name = 'NCGR', df2name = 'DIB_combined', ymax = 1, ymin = 0, ypos = 0.95):
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(6,6)
    outdf, fig, ax = scatter_diff(df1, df2, column = column, 
                 fig = fig, ax = axs[0], df1name = df1name, df2name = df2name, 
                                  ymax = ymax, ymin = ymin, ypos = ypos)
    violin_split(outdf, df1name, df2name, fig, axs[1], ymin = ymin, ymax = ymax)
    fig.suptitle(col_title, fontsize = 'x-large', fontweight = 'bold')
    return outdf, fig, ax

sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True,
               inner="quart", palette={"Male": "b", "Female": "y"})
sns.despine(left=True)

tips

p_refs_with_CRBB, fig, ax = create_plots(SRA_v_combined,combined_v_SRA, 'p_refs_with_CRBB', 'Proportion of contigs with CRB-BLAST')

reference_coverage, fig, ax = create_plots(SRA_v_combined, combined_v_SRA, 'reference_coverage', 'Reference coverage'
                                           , ymax = 0.9, ypos = 0.9)

linguistic_complexity, fig, ax = create_plots(SRA_v_combined,combined_v_SRA,  'linguistic_complexity', 'Linguistic complexity', ymax=0.25, ypos=0.025)

mean_orf_percent, fig, ax = create_plots(SRA_v_combined,combined_v_SRA,  'mean_orf_percent', 'Mean ORF percent',ymax=100, ypos=0.5)

n_seqs, fig, ax = create_plots(SRA_v_combined, combined_v_SRA, 'n_seqs', 'Number of contigs',ymax=60000, ypos=55000)

p_refs_with_CRBB, fig, ax = create_plots(ncgr_v_combined, combined_v_ncgr,'p_refs_with_CRBB', 'Proportion of contigs with CRB-BLAST')

reference_coverage, fig, ax = create_plots(ncgr_v_combined, combined_v_ncgr, 'reference_coverage', 'Reference coverage'
                                           , ymax = 0.9, ypos = 0.9)

linguistic_complexity, fig, ax = create_plots(ncgr_v_combined, combined_v_ncgr,  'linguistic_complexity', 'Linguistic complexity', ymax=0.25, ypos=0.025)

mean_orf_percent, fig, ax = create_plots(ncgr_v_combined, combined_v_ncgr,  'mean_orf_percent', 'Mean ORF percent',ymax=100, ypos=0.5)

n_seqs, fig, ax = create_plots(ncgr_v_combined, combined_v_ncgr, 'n_seqs', 'Number of contigs',ymax=60000, ypos=55000)



