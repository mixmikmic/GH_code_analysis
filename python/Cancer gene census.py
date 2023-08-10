# import and set options
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import utils
import os

# load config file
config = utils.load_config('config.yaml')

# make output directory
if not os.path.exists('output'): os.mkdir('output')

# get the significant genes for each method
signif_dict = utils.fetch_significant_genes('example_data/pancan',  # data directory
                                            .1,  # q-value threshold
                                            config)
num_methods = len(signif_dict)

# read in the CGC genes
# only includes genes with evidence for small somatic variants
cgc_genes = utils.process_cgc('CGC-freeze-download-date-20160401.tsv')

def num_cgc_overlap(signif_genes, cgc_list):
    """Intersect significant genes with CGC or other driver gene list."""
    cgc_result = {}
    for method in signif_genes:
        intersect = len(set(signif_genes[method]) & set(cgc_list))
        cgc_result[method] = intersect
    return cgc_result

# count the overlap
num_cgc_dict = num_cgc_overlap(signif_dict, cgc_genes)
num_signif_dict = {k: len(signif_dict[k]) for k in signif_dict}

# format result
overlap_df = pd.DataFrame({'# CGC': pd.Series(num_cgc_dict),
                           '# significant': pd.Series(num_signif_dict)})
overlap_df['Fraction overlap w/ CGC'] = overlap_df['# CGC'].astype(float) / overlap_df['# significant']
overlap_df

def plot_cgc_overlap(cgc_overlap_df, list_name='CGC', custom_order=None):
    """Create a bar plot for the fraction overlap with the cancer gene census (CGC).
    
    Parameters
    ----------
    cgc_overlap_df : pd.DataFrame
        Dataframe containing method names as an index and columns for '# CGC' and
        'Fraction overlap w/ CGC'
    custom_order : list or None
        Order in which the methods will appear on the bar plot
    """
    # Function to label bars
    def autolabel(rects):
        # attach some text labels
        for ii, rect in enumerate(rects):
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., height+.005, '%s' % (name[ii]),
                     ha='center', va='bottom', size=16)

    # order methods if no order given
    if custom_order is None:
        custom_order = cgc_overlap_df.sort_values('Fraction overlap w/ {0}'.format(list_name)).index.tolist()
            
    # make barplot
    name = cgc_overlap_df.ix[custom_order]['# '+list_name].tolist()
    with sns.axes_style('ticks'), sns.plotting_context('talk', font_scale=1.5):
        ax = sns.barplot(cgc_overlap_df.index,
                         cgc_overlap_df['Fraction overlap w/ {0}'.format(list_name)],
                         order=custom_order, color='black')

        # label each bar
        autolabel(ax.patches)

        # fiddle with formatting
        ax.set_xlabel('Methods')
        ax.set_ylabel('Fraction of predicted drivers\nfound in '+list_name)
        sns.despine()
        plt.xticks(rotation=45, ha='right', va='top')
        plt.gcf().set_size_inches(7, 7)
        # change tick padding
        plt.gca().tick_params(axis='x', which='major', pad=0)

    # format layout
    plt.tight_layout()

# make bar plot
order = ['ActiveDriver', 'OncodriveFM', 'OncodriveFML', 
         'OncodriveClust', 'MuSiC', 'TUSON', 'MutsigCV', '2020+']
plot_cgc_overlap(overlap_df, custom_order=order)

