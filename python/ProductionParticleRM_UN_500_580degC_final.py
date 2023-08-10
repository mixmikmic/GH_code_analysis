get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from scipy import stats
from sys import platform
from io import BytesIO

if platform == 'win32':
    path = 'C:/Users/ma.duerr/sciebo/Reference\ Particle\ Production/ParticleBrowser/'

elif platform == 'darwin':
    path = '/Users/mduerr/sciebo/Reference Particle Production/ParticleBrowser/'

# Path for the Secondary Electron EDAX PA search

directory_SE = 'PA_SE/'
file_500_SE = 'stub02/stub02.csv'
file_580_SE = 'stub01/stub01.csv'

# Path for the Backscatter EDAX PA search

directory_BS = 'PA_BS/'
file_500_BS = 'stub01/stub01.csv'
file_580_BS = 'stub02/stub02.csv'

# Pixel Size of the images
pixelsize = 0.11

def import_stubinfo(file):
    # gets the data from the .csv file generated from EDAX PA search 
    # returns a Pandas DataFrame
    #
    # The stream is opened as text io (universal newline mode) which converts the newlines to \n
    with open(file,'r') as f:
        inp_data = f.read()

        # extract the PA data
        import_stub = np.genfromtxt(BytesIO(inp_data.encode()), 
                                    delimiter=",", skip_header = 14, 
                                    names = True, autostrip=True, comments='//')
    return pd.DataFrame(import_stub)

df_500_SE = import_stubinfo(path + directory_SE + file_500_SE)
df_580_SE = import_stubinfo(path + directory_SE + file_580_SE)
df_500_BS = import_stubinfo(path + directory_BS + file_500_BS)
df_580_BS = import_stubinfo(path + directory_BS + file_580_BS)

df_500_BS.columns

IJ_file_500_SE = 'IJ_PA_SE_stub02_redirect.csv'
IJ_file_580_SE = 'IJ_PA_SE_stub01_redirect.csv'
IJ_file_500_BS = 'IJ_PA_BS_stub01_redirect.csv'
IJ_file_580_BS = 'IJ_PA_BS_stub02_redirect.csv'

def import_IJfile(file):
    # The stream is opened as text io (universal newline mode) which converts the newlines to \n
    with open(file,'r') as f:
        inp_data = f.read()

    # extract the PA data
    import_stub = np.genfromtxt(BytesIO(inp_data.encode()), 
                                delimiter=",", skip_header = 0, 
                                names = True, autostrip=True, comments='//')

    return pd.DataFrame(import_stub)

df_IJ500_SE = import_IJfile(path + directory_SE + IJ_file_500_SE)
df_IJ580_SE = import_IJfile(path + directory_SE + IJ_file_580_SE).sample(df_IJ500_SE.shape[0])
df_IJ500_BS = import_IJfile(path + directory_BS + IJ_file_500_BS)
df_IJ580_BS = import_IJfile(path + directory_BS + IJ_file_580_BS).sample(df_IJ500_BS.shape[0])

# The AvgDiam of the IJ Data in micrometer 
df_IJ500_SE['AvgDiam'] = df_IJ500_SE.loc(axis=1)['Major', 'Minor'].mean(1) * pixelsize

df_IJ580_SE['AvgDiam'] = df_IJ580_SE.loc(axis=1)['Major', 'Minor'].mean(1) * pixelsize

df_IJ500_BS['AvgDiam'] = df_IJ500_BS.loc(axis=1)['Major', 'Minor'].mean(1) * pixelsize

df_IJ580_BS['AvgDiam'] = df_IJ580_BS.loc(axis=1)['Major', 'Minor'].mean(1) * pixelsize

def plot_hist(data1, data2):
    # Plot histograms
    n, bins, patches = plt.hist([data1, data2],
                                bins = 60,
                                range = (0.75,2.5),
                                normed = 1,
                                alpha = 0.5,
                                histtype = 'stepfilled'
                               )

def check_monodispersity(diam_data, percent = 0.05):
    # Check monodispersity
    #particle count within 5% of median value
    within_percentile =((diam_data > (1-percent)*np.median(diam_data)) 
                   & (diam_data < (1+percent)*np.median(diam_data))).sum() 

    return within_percentile/ diam_data.size

hist_params = {
        'bins': 60,
        'range': (0,2.5),
        'normed': 1,
        'alpha': 0.5,
        'histtype': 'stepfilled'
}

def addmaxlines(frame, pos):
    frame.plot([pos,pos],[0,1], color ='black', alpha = 0.5, linewidth=1.0, linestyle="-")

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

# SE 500 deg C

n = ax0.hist([df_500_SE.AvgDiam.dropna().values,
         df_IJ500_SE.AvgDiam.dropna().values],
         label = ['Autom. PA', 'ImageJ'],
         **hist_params)

# find the modal of the distribution
#modal, count = stats.mode(df_500_SE.AvgDiam.dropna().values)
#add a line in the plot for modal and modal * power(2,1./3)
addmaxlines(ax0, np.median(df_500_SE.AvgDiam.dropna().values))
addmaxlines(ax0, np.median(df_500_SE.AvgDiam.dropna().values) * np.power(2,1./3))


ax0.set_title('SE 500 deg C')
ax0.legend()


# BS 500 deg C

ax1.hist([df_500_BS.AvgDiam.dropna().values,
         df_IJ500_BS.AvgDiam.dropna().values],
         **hist_params)
ax1.set_title('BS 500 deg C')

# SE 580 deg C

ax2.hist([df_580_SE[df_580_SE.UM > 0].AvgDiam.values,
         df_IJ580_SE.AvgDiam.dropna().values],
         **hist_params)
ax2.set_title('SE 580 deg C')

# BS 580 deg C

ax3.hist([df_580_BS.AvgDiam.dropna().values,
         df_IJ580_BS.AvgDiam.dropna().values],
         **hist_params)
ax3.set_title('BS 580 deg C')

#add a line in the plot for modal and modal * power(2,1./3)
addmaxlines(ax3, np.median(df_IJ580_SE.AvgDiam.dropna().values))
addmaxlines(ax3, np.median(df_IJ580_SE.AvgDiam.dropna().values) * np.power(2,1./3))

# axes labels, ticks, etc...


fig.subplots_adjust(hspace = 0.2, wspace = 0.1)
fig.suptitle('Size distribution - Comparison between PA analysis methods', fontsize = 18)
fig.set_figheight(8)
fig.set_figwidth(10)

data1 = df_500_SE.AvgDiam.dropna().values
data2 = df_500_SE[df_500_SE.SiK > 0].AvgDiam.dropna().values
data3 = df_500_SE[(df_500_SE.UM > 0)].AvgDiam.dropna().values
plt.hist([data1, data2, data3],
         label = ['df_500_SE', 'with EDX','with UM > 0'],
         bins = 60,
         range = (0,2.5),
         normed =  1,
         alpha = 0.5,
         histtype = 'step',
         log = True
        )
plt.legend()
plt.show()

# Fontsize

fs = 12

hist_params = {
        'bins': 60,
        'range': (0,2.5),
        'normed': 0,
        'alpha': 0.5,
        'histtype': 'stepfilled'
}

legend_params = {
    'loc': 'best',
    'fancybox' : True,
    'framealpha' : 0.5,
    'fontsize' : 10    
}

fig, axes = plt.subplots(nrows=1, ncols=2)
ax0, ax1 = axes.flatten()

# IJ 500 deg C

ax0.hist([df_IJ500_SE.AvgDiam.dropna().values,
         df_IJ500_BS.AvgDiam.dropna().values],
         label = ['SE', 'BS'],
         **hist_params)
ax0.set_title('Image J 500 deg C')
ax0.legend(** legend_params)

# IJ 580 deg C

ax1.hist([df_IJ580_SE.AvgDiam.dropna().values,
         df_IJ580_BS.AvgDiam.dropna().values],
         label = ['SE', 'BS'],
         **hist_params)
ax1.set_title('Image J 580 deg C', fontsize = fs)
ax1.legend(** legend_params)

# axes labels, ticks, etc...
for ax in axes:
    ax.set_xlabel('Average Diameter / µm', fontsize = fs)
    ax.set_ylabel('Particle Count', fontsize = fs)

fig.subplots_adjust(hspace = 0.2, wspace = 0.2)
fig.suptitle('Size distribution - Comparison between SE and BE images', fontsize = 18, y = 1.05)
fig.set_figheight(4)
fig.set_figwidth(10)

#fontsize
fs = 14

data1 = df_IJ500_SE.AvgDiam.dropna().values
data2 = df_IJ580_SE.AvgDiam.dropna().values

figp, ax = plt.subplots(1)

ml = MultipleLocator(5)

ax.hist([data1, data2],
         label = ['T = 500$^\circ$C', 'T = 580$^\circ$C'],
         bins = 100,
         range = (0.5,2.5),
         normed =  0,
         alpha = 0.5,
         histtype = 'stepfilled',
         lw = 1.5
        )
# axes labels, ticks, etc...

ax.set_xlabel('Average Diameter / µm', fontsize=fs)
ax.set_ylabel('Particle Count', fontsize=fs)
ax.set_xticks(np.arange(0.5,2.5,0.5))
ax.xaxis.set_minor_locator(ml)
ax.tick_params(axis='x', which = 'minor', bottom= 'on', top = 'on', 
               direction = 'in', width =1, length = 3)
ax.tick_params(axis='x', which = 'major', width=1, length = 6, labelsize = 12)

ax.tick_params(axis='y', which = 'minor', left= 'off', right = 'off')
ax.tick_params(axis='y', which = 'major', direction='in', width=1, length = 3, labelsize = 12)

ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize = 12)
plt.minorticks_on()

plt.figure(figsize = (3.5,2.5))
figp.savefig(path+directory_SE+'size_distr_SE.png',dpi=600)
plt.show()

df_IJ580_SE.shape

np.median(df_IJ580_SE.AvgDiam.dropna().values) * np.power(2,1./3)

np.median(df_IJ580_SE.AvgDiam[(df_IJ580_SE.AvgDiam > 1.4) & (df_IJ580_SE.AvgDiam < 1.6)])

# the bins to categorize the AvgDiam (µm)
d_0, d_1, d_2, d_3, d_4 = 0, 1.0 ,1.4, 1.6, 5

# the bins to categorize the Circ
c_0, c_1, c_2 = 0, 0.75, 1

# use pandas categorical index and pd.cut function
# the four (4) size categories of AvgDiam are
# debris, SD: single droplet, DD: double droplet, L: large particles
#
# the two (2) circularity categories are
# round and irregular

labels_size = ['debris', 'SD', 'DD', 'L']
labels_circularity = ['irregular', 'round']

df_IJ500_SE['sizecategory']=pd.cut(df_IJ500_SE.AvgDiam, (d_0,d_1,d_2,d_3,d_4),
       labels = labels_size
      )

df_IJ580_SE['sizecategory']=pd.cut(df_IJ580_SE.AvgDiam, (d_0,d_1,d_2,d_3,d_4),
       labels = labels_size
      )

df_IJ500_SE['circcategory']=pd.cut(df_IJ500_SE.Circ, (c_0, c_1, c_2),
       labels = labels_circularity
      )

df_IJ580_SE['circcategory']=pd.cut(df_IJ580_SE.Circ, (c_0, c_1, c_2),
       labels = labels_circularity
      )

hist_params = {
        'bins': 250,
        'range': (0,5),
        'normed': 0,
        'alpha': 0.5,
        'histtype': 'stepfilled'
}

scatter_params = {
        's' : 2,
        'alpha' : 0.5,
        'color' : 'royalblue'
}

patch_params = {
        'facecolor' : 'none',
}

text_params = {
        'fontsize' : 18,
        'weight': 'bold',
        'alpha' : 0.6
}

ml = MultipleLocator(10)

x1 = df_IJ500_SE.AvgDiam.dropna().values
y1 = df_IJ500_SE.Circ.dropna().values

x2 = df_IJ580_SE.AvgDiam.dropna().values
y2 = df_IJ580_SE.Circ.dropna().values

fig, axes = plt.subplots(nrows=2, ncols=2)

ax0, ax1, ax2, ax3 = axes.flatten()

# Plot the histograms
ax0.hist(x1,**hist_params)

ax1.hist(x2,**hist_params)

ax2.scatter(x1,y1, **scatter_params)

ax3.scatter(x2,y2, **scatter_params)


for axis in [ax2, ax3]:
    # patch cat I
    axis.add_patch(patches.Rectangle((d_1,c_1), d_2-d_1, c_2-c_1, **patch_params))

    # patch cat II
    axis.add_patch(patches.Rectangle((d_2,c_1), d_3-d_2, c_2-c_1, **patch_params))

    # patch cat III
    axis.add_patch(patches.Rectangle((d_3,c_1), d_4-d_3, c_1-c_0, **patch_params))    
    
    # patch cat IV    
    axis.add_patch(patches.Rectangle((d_3,c_0), d_4-d_3, c_1-c_0, **patch_params))  
    
    axis.set_xticks(np.arange(0,5,1))
    axis.xaxis.set_minor_locator(ml)
    axis.tick_params(axis='x', which = 'minor', bottom= 'on', top = 'on', 
                   direction = 'in', width =0.75, length = 3)
    axis.tick_params(axis='x', which = 'major', width=0.75, length = 3)

    axis.tick_params(axis='y', which = 'minor', left= 'off', right = 'off')
    axis.tick_params(axis='y', which = 'major', direction='in', width=1, length = 3)
    
    axis.set_xlim(0,5)
    axis.set_ylim(0,1)
    
    axis.set_xlabel('Average Diameter / µm', fontsize=fs)
    axis.set_ylabel('Circularity', fontsize=fs)
    
    # Add the annotation of the four categories
    axis.text(0.8,0.5, 'I', **text_params)
    axis.text(1.2,0.5, 'II', **text_params)
    axis.text(3,0.9, 'III', **text_params)
    axis.text(3,0.5, 'IV', **text_params)
    axis.arrow(0.9, 0.58, 0.25, 0.15, alpha = 0.6)
    axis.arrow(1.4, 0.58, 0.1, 0.15, alpha = 0.6)

ax0.set_title('T = 500$^\circ$C', fontsize = fs)
ax1.set_title('T = 580$^\circ$C', fontsize = fs)   



# hide the x ticks for top plots and the y ticks 
plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axes[0, :]], visible=False)

fig.subplots_adjust(hspace=0) 

fig.set_figheight(8)
fig.set_figwidth(10)

#plt.savefig(path+directory_SE + 'shape_categories.png',dpi=600)

stat_500 = df_IJ500_SE.groupby(['sizecategory', 'circcategory']).describe().AvgDiam
stat_580 = df_IJ580_SE.groupby(['sizecategory', 'circcategory']).describe().AvgDiam

# grab the counts of the categorized dataframe
count_500 = stat_500.loc[(slice(None)), (slice(None)),'count']
count_580 = stat_580.loc[(slice(None)), (slice(None)),'count']

#Create a dictionary of category statistics
cat_index = ['I', 'II', 'III', 'IV']
cat_stat = pd.DataFrame(
            {'500 deg' : pd.Series([
                                #cat_i 
                                count_500.loc['SD','round'],
                                #cat_ii
                                count_500.loc['DD','round'],
                                #cat_iii
                                count_500.loc['L','round'],
                                #cat_iv
                                count_500.loc['L','irregular'],                                
                                ],
                            index = cat_index
            ),
            '580 deg' : pd.Series([
                                #cat_i 
                                count_580.loc['SD','round'],
                                #cat_ii
                                count_580.loc['DD','round'],
                                #cat_iii
                                count_580.loc['L','round'],
                                #cat_iv
                                count_580.loc['L','irregular']
                                ],
                            index = cat_index
            )
           }
)

# Calculate the Percentage and add as column
cat_stat['500 deg_perc'] = cat_stat['500 deg'] / cat_stat['500 deg'].sum() *100
cat_stat['580 deg_perc'] = cat_stat['580 deg'] / cat_stat['580 deg'].sum() *100

cat_stat

import skimage
from skimage import io

def loadthumbs_selection(part_ids, thumb_path, *ext):
    # loads the images from thumbpaths with the part_ids
    # returns an np.array of concatenated images
    
    def make_pathsfromlistentry(directory, listentry, extension):
        return directory + '{:0>5d}'.format(listentry) + extension

    # create the file list        
    filelist = [make_pathsfromlistentry(thumb_path, int(i), '.png') for i in part_ids]
    
    # load images into a scikit.io imagecollection
    coll = io.imread_collection(filelist)
    
    # convert the collection into a list of images and concatenate into a np.array    
    allimgs = np.concatenate([img for img in coll], axis = 1)
    
    return allimgs

thumbpath = path + directory_SE + 'stub02/IJselect/'

particles_cati = df_IJ500_SE[(df_IJ500_SE.sizecategory == 'SD') & (df_IJ500_SE.circcategory == 'round')].sample(20).id.values

particles_catii = df_IJ500_SE[(df_IJ500_SE.sizecategory == 'DD') & (df_IJ500_SE.circcategory == 'round')].sample(20).id.values

particles_catiii = df_IJ500_SE[(df_IJ500_SE.sizecategory == 'L') & (df_IJ500_SE.circcategory == 'round')].sample(20).id.values

particles_cativ = df_IJ500_SE[(df_IJ500_SE.sizecategory == 'L') & (df_IJ500_SE.circcategory == 'irregular')].sample(20).id.values

# grab the images

thumbs_cati = loadthumbs_selection(particles_cati, thumbpath)

thumbs_catii = loadthumbs_selection(particles_catii, thumbpath)

thumbs_catiii = loadthumbs_selection(particles_catiii, thumbpath)

thumbs_cativ = loadthumbs_selection(particles_cativ, thumbpath)

io.imshow(thumbs_cati)

#io.imsave(path + directory_SE + 'SE_500_thumbs_cati.png',thumbs_cati)

io.imshow(thumbs_catii)

#io.imsave(path + directory_SE + 'SE_500_thumbs_catii.png',thumbs_catii)

io.imshow(thumbs_catiii)

#io.imsave(path + directory_SE + 'SE_500_thumbs_catiii.png',thumbs_catiii)

io.imshow(thumbs_cativ)

#io.imsave(path + directory_SE + 'SE_500_thumbs_cativ.png',thumbs_cativ)

thumbpath = path + directory_SE + 'stub01/IJselect/'

particles_cati = df_IJ580_SE[(df_IJ580_SE.sizecategory == 'SD') & (df_IJ580_SE.circcategory == 'round')].sample(20).id.values

particles_catii = df_IJ580_SE[(df_IJ580_SE.sizecategory == 'DD') & (df_IJ580_SE.circcategory == 'round')].sample(20).id.values

particles_catiii = df_IJ580_SE[(df_IJ580_SE.sizecategory == 'L') & (df_IJ580_SE.circcategory == 'round')].sample(20).id.values

particles_cativ = df_IJ580_SE[(df_IJ580_SE.sizecategory == 'L') & (df_IJ580_SE.circcategory == 'irregular')].sample(20).id.values

# grab the images

thumbs_cati = loadthumbs_selection(particles_cati, thumbpath)

thumbs_catii = loadthumbs_selection(particles_catii, thumbpath)

thumbs_catiii = loadthumbs_selection(particles_catiii, thumbpath)

thumbs_cativ = loadthumbs_selection(particles_cativ, thumbpath)

io.imshow(thumbs_cati)

#io.imsave(path + directory_SE + 'SE_580_thumbs_cati.png',thumbs_cati)

io.imshow(thumbs_catii)

#io.imsave(path + directory_SE + 'SE_580_thumbs_catii.png',thumbs_catii)

io.imshow(thumbs_catiii)

#io.imsave(path + directory_SE + 'SE_580_thumbs_catiii.png',thumbs_catiii)

io.imshow(thumbs_cativ)

#io.imsave(path + directory_SE + 'SE_580_thumbs_cativ.png',thumbs_cativ)



