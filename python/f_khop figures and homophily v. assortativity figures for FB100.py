import pandas as pd
from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
import os

## relevant libraries
execfile('../functions/python_libraries.py')
execfile('../functions/compute_homophily.py')
execfile('../functions/compute_monophily.py')

# dataset created from soal_script_facebook_script_homophily_index_vs_Newmans_assortativity.py
homophily_assortativity_df = pd.read_csv('../../data/output/facebook_homophily_vs_newmans_assortativity_Dec2017.csv')
drop_schools = np.array(['Wellesley22', 'Smith60', 'Simmons81'])
homophily_assortativity_df = homophily_assortativity_df.loc[~np.in1d(homophily_assortativity_df.school ,drop_schools)]
homophily_assortativity_df.head()

get_ipython().magic('matplotlib inline')
ax = plt.subplot(111)

ax.scatter(homophily_assortativity_df.cc_avg_homophily,
           homophily_assortativity_df.cc_gender_assortativity, 
          color = 'black', alpha = 0.6)


ax.set_xlabel('Homophily Index (class-averaged)')
ax.set_ylabel('Assortativity')

ax.set_xlim(0.44,0.62)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.annotate('FB100 School', xy=(0.445, 0.18), 
                 color='black', alpha=1, size=12)

#plt.show()

pp = PdfPages('../../figures/homophily_index_vs_assortativity_fb100.pdf')
pp.savefig()
pp.close()

#dataset created by: compare_k_hop_friends_vs_AUC_fb.py
khop = pd.read_csv('../../data/output/khop_vs_auc_Nov2017.csv')
khop.head()

drop_schools = np.array(['Wellesley22', 'Smith60', 'Simmons81'])

khop = khop.loc[~np.in1d(khop.school ,drop_schools)]

schools = np.unique(khop.school)

get_ipython().magic('matplotlib inline')
from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

fig = plt.figure()
seaborn.set_style(style='white')
from mpl_toolkits.axes_grid1 import Grid
grid = Grid(fig, rect=111, nrows_ncols=(1,1),
            axes_pad=0.1, label_mode='L')
for i in range(4):
    if i == 0:
        grid[i].xaxis.set_major_locator(FixedLocator([1,2,3,4,5]))
        grid[i].yaxis.set_major_locator(FixedLocator([0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1]))

        for j in range(len(schools)):
            auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school==schools[j]]
            k_hop = np.array(range(len(auc_array)))+1
            grid[i].plot(k_hop,
               auc_array, color = 'gray', alpha = 0.3)


        auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school=='Amherst41']
        grid[i].plot(k_hop,
           auc_array, color = 'black')#, alpha = 0.3)
        grid[i].scatter(k_hop,
           auc_array, color = 'black')#, alpha = 0.5)

        grid[i].set_xlim(0.9,5.1)
        grid[i].set_ylim(0.25,1.01)

        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].tick_params(axis='both', which='major', labelsize=13)
        grid[i].tick_params(axis='both', which='minor', labelsize=13)
        grid[i].set_xlabel('Neighborhood Distance from Ego Node')
        grid[i].set_ylabel('AUC at Neighborhood Distance')
        grid[i].annotate('Amherst College', xy=(1.07, 1), 
                 color='black', alpha=1, size=12)
        grid[i].annotate('Other FB100 Schools', xy=(1.07, 0.96), 
                 color='gray', alpha=1, size=12)

grid[0].set_xticks([1,2,3,4,5])
grid[0].set_yticks([ 0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1])


grid[0].minorticks_on()
grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
pp = PdfPages('../../figures/khop_vs_auc_NHB_figure_python.pdf')
pp.savefig()
pp.close()
plt.show()

get_ipython().magic('matplotlib inline')
from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

fig = plt.figure(figsize=(6.69291/2,2.5), dpi = 300)
text_size = 8
axis_text = 8
tick_label_size = 7
msize = 5
pad = 0#.1

seaborn.set_style(style='white')
from mpl_toolkits.axes_grid1 import Grid
grid = Grid(fig, rect=111, nrows_ncols=(1,1),
            axes_pad=1, label_mode='L')
for i in range(4):
    if i == 0:
        grid[i].xaxis.set_major_locator(FixedLocator([1,2,3,4,5]))
        grid[i].yaxis.set_major_locator(FixedLocator([0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1]))

        for j in range(len(schools)):
            auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school==schools[j]]
            k_hop = np.array(range(len(auc_array)))+1
            grid[i].plot(k_hop,
               auc_array, color = 'gray', alpha = 0.4)


        auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school=='Amherst41']
        grid[i].plot(k_hop,
           auc_array, color = 'black')#, alpha = 0.3)
        #grid[i].scatter(k_hop,
        #   auc_array, color = 'black')#, alpha = 0.5)

        grid[i].set_xlim(0.9,5.1)
        grid[i].set_ylim(0.25,1.05)
        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].spines["left"].set_linewidth(0.75)
        grid[i].spines["bottom"].set_linewidth(0.75)
        grid[i].tick_params(axis='both', which='major', labelsize=axis_text,
                           length = 0.1,
                           width = 0.5)
        grid[i].tick_params(axis='both', which='minor', labelsize=axis_text)
        grid[i].set_xlabel('Neighborhood Distance from Ego Node',
                            size = axis_text,labelpad = pad)
        grid[i].set_ylabel('AUC at Neighborhood Distance',
                           size = axis_text,labelpad = pad)
        grid[i].annotate('Amherst College', xy=(1.07, 1), 
                 color='black', alpha=1, size=text_size)
        grid[i].annotate('Other FB100 Schools', xy=(1.07, 0.96), 
                 color='gray', alpha=1, size=text_size)
grid[0].set_xticks([1,2,3,4,5])
grid[0].set_yticks([ 0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1])
grid[0].minorticks_on()
grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
pp = PdfPages('../../figures/khop_vs_auc_NHB_figure_python_COLUMN.pdf')
pp.savefig(dpi = 300)
pp.close()
plt.show()







get_ipython().magic('matplotlib inline')
from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

fig = plt.figure()
seaborn.set_style(style='white')
from mpl_toolkits.axes_grid1 import Grid
grid = Grid(fig, rect=111, nrows_ncols=(1,1),
            axes_pad=0.1, label_mode='L')
for i in range(4):
    if i == 0:
        grid[i].xaxis.set_major_locator(FixedLocator([1,2,3,4,5]))
        grid[i].yaxis.set_major_locator(FixedLocator([0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1]))

        for j in range(len(schools)):
            auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school==schools[j]]
            k_hop = np.array(range(len(auc_array)))+1
            grid[i].plot(k_hop,
               auc_array, color = 'gray', alpha = 0.3)


        auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school=='Amherst41']
        grid[i].plot(k_hop,
           auc_array, color = 'black')#, alpha = 0.3)
        grid[i].scatter(k_hop,
           auc_array, color = 'black')#, alpha = 0.5)
        auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school=='MIT8']
        grid[i].plot(k_hop,
           auc_array, color = 'darkblue')#, alpha = 0.3)
        grid[i].scatter(k_hop,
           auc_array, color = 'darkblue')#, alpha = 0.5)
        
        
        grid[i].set_xlim(0.9,5.1)
        grid[i].set_ylim(0.25,1.01)

        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].tick_params(axis='both', which='major', labelsize=13)
        grid[i].tick_params(axis='both', which='minor', labelsize=13)
        grid[i].set_xlabel('Neighborhood Distance from Ego Node')
        grid[i].set_ylabel('AUC at Neighborhood Distance')
        grid[i].annotate('Amherst College', xy=(3.5, 1), 
                 color='black', alpha=1, size=12)
        grid[i].annotate('Other FB100 Schools', xy=(3.5, 0.92), 
                 color='gray', alpha=1, size=12)
        grid[i].annotate('MIT', xy=(3.5, 0.96), 
                 color='darkblue', alpha=1, size=12)

grid[0].set_xticks([1,2,3,4,5])
grid[0].set_yticks([ 0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1])


grid[0].minorticks_on()
grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
pp = PdfPages('../../figures/khop_vs_auc_NHB_figure_python_SI.pdf')
pp.savefig()
pp.close()
plt.show()

## dataset created from: compare_k_hop_friends_vs_AUC_add_health.py
khop = pd.read_csv('../../data/output/khop_vs_auc_add_health_undirected_Nov2017.csv')
khop.head()

schools = np.unique(khop.school)
#print len(schools)

school_drop = np.array(['comm27'])
khop = khop[~np.in1d(khop.school,school_drop)] #%in% schools

get_ipython().magic('matplotlib inline')
from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

fig = plt.figure()
seaborn.set_style(style='white')
from mpl_toolkits.axes_grid1 import Grid
grid = Grid(fig, rect=111, nrows_ncols=(1,1),
            axes_pad=0.1, label_mode='L')
for i in range(4):
    if i == 0:
        grid[i].xaxis.set_major_locator(FixedLocator([1,2,3,4,5]))
        grid[i].yaxis.set_major_locator(FixedLocator([0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1]))

        for j in range(len(schools)):
            auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school==schools[j]]
            k_hop = np.array(range(len(auc_array)))+1
            grid[i].plot(k_hop,
               auc_array, color = 'gray', alpha = 0.3)


        #auc_array = np.array(khop.auc_in_khop_neighborhood)[khop.school=='Amherst41']
        #grid[i].plot(k_hop,
        #   auc_array, color = 'black')#, alpha = 0.3)
        #grid[i].scatter(k_hop,
        #   auc_array, color = 'black')#, alpha = 0.5)

        grid[i].set_xlim(0.9,5.1)
        grid[i].set_ylim(0.1,1.01)

        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].tick_params(axis='both', which='major', labelsize=13)
        grid[i].tick_params(axis='both', which='minor', labelsize=13)
        grid[i].set_xlabel('Neighborhood Distance from Ego Node')
        grid[i].set_ylabel('AUC at Neighborhood Distance')
        #grid[i].annotate('Amherst College', xy=(1.07, 1), 
        #         color='black', alpha=1, size=12)
        #grid[i].annotate('Other FB100 Schools', xy=(1.07, 0.96), 
        #         color='gray', alpha=1, size=12)

grid[0].set_xticks([1,2,3,4,5])
grid[0].set_yticks([ 0,0.1,0.2, 0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1])


grid[0].minorticks_on()
grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
pp = PdfPages('../../figures/khop_vs_auc_NHB_figure_python_add_health.pdf')
pp.savefig()
pp.close()
plt.show()

## dataset for FB100 created for compare_k_hop_friends_vs_proportion_class_same.py
khop = pd.read_csv('../../data/output/khop_vs_proportion_same_Nov2017.csv')
print khop.head()
khop = khop.loc[~np.in1d(khop.school ,drop_schools)]
schools = np.unique(khop.school)

get_ipython().magic('matplotlib inline')
ax = plt.subplot(111)

for j in range(len(schools)):
    auc_array = np.array(khop.proportion_nodes_majority_same_class_in_khop_neighborhood)[khop.school==schools[j]]
    k_hop = np.array(range(len(auc_array)))+1
    ax.plot(k_hop,
           auc_array, color = 'gray', alpha = 0.3)
auc_array = np.array(khop.proportion_nodes_majority_same_class_in_khop_neighborhood)[khop.school=='Amherst41']
ax.plot(k_hop,
           auc_array, color = 'black', alpha = 1)
ax.scatter(k_hop,
           auc_array, color = 'black', alpha = 1, s=20)


auc_array = np.array(khop.proportion_nodes_majority_same_class_in_khop_neighborhood)[khop.school=='MIT8']
ax.plot(k_hop,
           auc_array, color = 'darkblue', alpha = 1)
ax.scatter(k_hop,
           auc_array, color = 'darkblue', alpha = 1, s=20)


ax.set_xlabel('Neighborhood Distance from Ego Node')
ax.set_ylabel('Proportion Same Neighbors')

ax.set_xlim(0.9,5.1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_ylim(-100,100)

ax.annotate('Amherst College', xy=(3.5, 0.88), 
                 color='black', alpha=1, size=12)
ax.annotate('MIT', xy=(3.5, 0.84), 
                 color='darkblue', alpha=1, size=12)
ax.annotate('Other FB100 Schools', xy=(3.5, 0.80), 
                 color='gray', alpha=1, size=12)
pp = PdfPages('../../figures/khop_vs_accuracy_proportion_same_NHB_figure_python.pdf')
pp.savefig()
pp.close()

## implemented LINK with solver='lbfgs'
from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
import os


## relevant libraries
execfile('../functions/python_libraries.py')

## processing datasets
execfile('../functions/create_adjacency_matrix.py') 
execfile('../functions/create_directed_adjacency_matrix.py')


execfile('../functions/parsing.py')
execfile('../functions/mixing.py')

## code for gender prediction 
execfile('../functions/LINK.py')
execfile('../functions/majority_vote.py')
execfile('../functions/SI_functions/majority_vote_modified_SI.py')
execfile('../functions/compute_chi_square.py')

execfile('../functions/ZGL.py')
execfile('../functions/benchmark_classifier.py')

## gender preference distribution
execfile('../functions/compute_null_distribution.py')


## filename where relevant FB100 data is stored
fb100_file = '/Users/kristen/Dropbox/gender_graph_data/FB_processing_pipeline/data/0_original/'

school = 'MIT8' #MIT8 Amherst41
for f in listdir(fb100_file):
    if f.endswith('.mat'):
        tag = f.replace('.mat', '')
        if (tag == school):
            print tag
            input_file = path_join(fb100_file, f)
            A, metadata = parse_fb100_mat_file(input_file)

            adj_matrix_tmp = A.todense()
            gender_y_tmp = metadata[:,1] #gender
                
            gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)
                
            (gender_y, adj_matrix_gender) = create_adj_membership(
                                    nx.from_numpy_matrix(adj_matrix_tmp), # graph
                                                           gender_dict,   # dictionary
                                                           0,             # val_to_drop, gender = 0 is missing
                                                           'yes',         # delete_na_cols, ie completely remove NA nodes from graph
                                                           0,             # diagonal
                                                           None,          # directed_type
                                                           'gender')      # gender
            
            gender_y = np.array(map(np.int,gender_y)) ## need np.int for machine precisions reasons

F_fb_label = 1
M_fb_label = 2

percent_initially_unlabelled = [0.99,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]#,0.01]
percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)

n_iter = 100

(mean_accuracy_mv_amherst, se_accuracy_mv_amherst, 
 mean_micro_auc_mv_amherst,se_micro_auc_mv_amherst, 
 mean_wt_auc_mv_amherst,se_wt_auc_mv_amherst) =majority_vote_modified(percent_initially_unlabelled,  
                                                                np.array(gender_y), 
                                                                np.array(adj_matrix_gender), 
                                                                       num_iter=n_iter)



adj_amherst2= np.matrix(adj_matrix_gender)**2
adj_amherst2[range(adj_amherst2.shape[0]),range(adj_amherst2.shape[0])]=0 ## remove self-loops

(mean_accuracy_mv2_amherst2, se_accuracy_mv2_amherst2, 
 mean_micro_auc_mv2_amherst2,se_micro_auc_mv2_amherst2, 
 mean_wt_auc_mv2_amherst2,se_wt_auc_mv2_amherst2) =majority_vote_modified(percent_initially_unlabelled,  
                                                                np.array(gender_y), 
                                                                np.array(adj_amherst2), 
                                                                num_iter=n_iter) 



get_ipython().magic('matplotlib inline')
from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

fig = plt.figure()
seaborn.set_style(style='white')
from mpl_toolkits.axes_grid1 import Grid
grid = Grid(fig, rect=111, nrows_ncols=(1,1),
            axes_pad=0.1, label_mode='L')
for i in range(4):
    if i == 0:
        grid[i].xaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
        grid[i].yaxis.set_major_locator(FixedLocator([0.4, 0.5,0.6,0.7,0.8,0.9,1]))

        grid[i].errorbar(percent_initially_labelled*100, mean_accuracy_mv_amherst,
            yerr=se_accuracy_mv_amherst, fmt='--o', capthick=2,
            alpha=1, elinewidth=3, color='red')
        grid[i].errorbar(percent_initially_labelled*100, mean_accuracy_mv2_amherst2, 
            yerr=se_accuracy_mv2_amherst2, fmt='--o', capthick=2,
                alpha=1, elinewidth=3, color='maroon')
        
        #grid[i].annotate('LINK', xy=(3, 0.99), 
        #         color='black', alpha=1, size=12)
        grid[i].annotate('2-hop MV', xy=(3, 0.96), 
                 color='maroon', alpha=1, size=12)
        grid[i].annotate('1-hop MV', xy=(3, 0.93), 
                 color='red', alpha=1, size=12)
        grid[i].set_xlim(0,100)
        grid[i].set_ylim(0.49,1.01)
        grid[i].set_xlim(0,100)
        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].tick_params(axis='both', which='major', labelsize=13)
        grid[i].tick_params(axis='both', which='minor', labelsize=13)
        grid[i].set_xlabel('Percent of Nodes Initially Labeled')
        grid[i].set_ylabel('Accuracy (macro F1-score)')

#plt.setp(ax1, xticks=[0,25, 50, 75, 100], xticklabels=['0', '25', '50', '75', '100'])
grid[0].set_yticks([ 0.4, 0.5,0.6,0.7,0.8,0.9,1])

grid[0].set_title(school)

grid[0].minorticks_on()
grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
#plt.show()

pp = PdfPages('../../figures/' + school +'_Inference_SI.pdf')
pp.savefig()
pp.close()

get_ipython().magic('matplotlib inline')
from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

fig = plt.figure()
seaborn.set_style(style='white')
from mpl_toolkits.axes_grid1 import Grid
grid = Grid(fig, rect=111, nrows_ncols=(1,1),
            axes_pad=0.1, label_mode='L')
for i in range(4):
    if i == 0:
        grid[i].xaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
        grid[i].yaxis.set_major_locator(FixedLocator([0.4, 0.5,0.6,0.7,0.8,0.9,1]))

        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_mv_amherst,
            yerr=se_micro_auc_mv_amherst, fmt='--o', capthick=2,
            alpha=1, elinewidth=3, color='red')
        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_mv2_amherst2, 
            yerr=se_micro_auc_mv2_amherst2, fmt='--o', capthick=2,
                alpha=1, elinewidth=3, color='maroon')
        
        #grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_baseline_amherst, 
        #    yerr=se_wt_auc_baseline_amherst, fmt='--o', capthick=2,
        #   alpha=1, elinewidth=3, color='gray')
        

        grid[i].annotate('2-hop MV', xy=(3, 0.96), 
                 color='maroon', alpha=1, size=12)
        grid[i].annotate('1-hop MV', xy=(3, 0.93), 
                 color='red', alpha=1, size=12)
        grid[i].set_ylim(0.49,1.01)
        grid[i].set_xlim(0,100)
        grid[i].spines['right'].set_visible(False)
        grid[i].spines['top'].set_visible(False)
        grid[i].tick_params(axis='both', which='major', labelsize=13)
        grid[i].tick_params(axis='both', which='minor', labelsize=13)
        grid[i].set_xlabel('Percent of Nodes Initially Labeled')
        grid[i].set_ylabel('AUC')

#plt.setp(ax1, xticks=[0,25, 50, 75, 100], xticklabels=['0', '25', '50', '75', '100'])
grid[0].set_xticks([0,25, 50, 75, 100])
grid[0].set_yticks([ 0.4, 0.5,0.6,0.7,0.8,0.9,1])

grid[0].set_title(school)

grid[0].minorticks_on()
grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
#plt.show()
pp = PdfPages('../../figures/' + school +'_AUC_Inference.pdf')
pp.savefig()
pp.close()



