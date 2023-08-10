import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
get_ipython().magic('matplotlib inline')
import sys
from IPython.display import display, clear_output
import os
sys.path.append('/Users/palmer/Documents/python_codebase/')
from pyIMS.hdf5.inMemoryIMS_hdf5 import inMemoryIMS_hdf5
from pyMS.pyisocalc import pyisocalc
import time
import pickle
from pySpatialMetabolomics.parse_databases import parse_databases

## Input parameters
# Data file/ouput location
filename_in =  '/Users/palmer/Documents/tmp_data/UCSD_01902_centroids_IMS.hdf5' #using a temporary hdf5 based format
output_dir = '' 

## Isotope pattern prediction
# possible adducts
adducts = ('H','Na','K')
charge = 1
# Database file locations
db_filename = '/Users/palmer/Copy/hmdb_database.tsv'
db_dump_folder = '/Users/palmer/Documents/Projects/2015/SM_development/databases' # store location for isotope patterns - to avoid regenerating every time
# Instrument prediction parameters
isocalc_sig=0.01
isocalc_resolution = 200000.
isocalc_do_centroid = True

## Image genreation parameters
ppm = 2.; # parts per million -  a measure of how accuracte the mass spectrometer is
nlevels = 30 # parameter for measure of chaos
q=99 # percentile threshold ()

## Results filter thresholds
measure_tol = 0.99#heuristic tolerances for filter
iso_corr_tol = 0.5
iso_ratio_tol = 0.85

## Load database of metabolites
# TODO: standardise database format
# currently a separate load function is needed each time

# sum_formula = 'C23H45NO4' #Glycocholic acid: http://89.223.39.196:2347/substance/00138
# sum_formulae = read_kegg_compounds(db_filename)
# sum_formulae = read_hmdb_compounds(db_filename)
sum_formulae = parse_databases.read_hmdb_compounds(db_filename)

## Predict isotope patterns
def calcualte_isotope_patterns(sum_formulae,adducts='',isocalc_sig=0.01,isocalc_resolution = 200000.,isocalc_do_centroid = True):
    tstart=time.time()
    mz_list={}
    for n,sum_formula in enumerate(sum_formulae):
        if np.mod(n,10) == 0:
            clear_output(wait=True)
            print '{} {:2.3f}\% complete\r'.format(sum_formula,100*float(n)/len(sum_formulae),end="\r")
            sys.stdout.flush()
        for adduct in adducts:        
            isotope_ms = pyisocalc.isodist(sum_formula+adduct,plot=False,sigma=isocalc_sig,charges=charge,resolution=isocalc_resolution,do_centroid=isocalc_do_centroid)
            if not sum_formula in mz_list:
                mz_list[sum_formula] = {}
            mz_list[sum_formula][adduct] = isotope_ms.get_spectrum(source='centroids')
    print 'Elapsed: {:5.2f} seconds'.format(time.time() - tstart)
    return mz_list

# Check if already genrated and load if possible
db_name=  os.path.splitext(os.path.basename(db_filename))[0]
mz_list={}
for adduct in adducts:
    load_file = '{}/{}_{}_{}_{}.dbasedump'.format(db_dump_folder,db_name,adduct,isocalc_sig,isocalc_resolution)
    if os.path.isfile(load_file):
        mz_list_tmp = pickle.load(open(load_file,'r'))
    else:
        mz_list_tmp = calcualte_isotope_patterns(sum_formulae,adducts=(adduct,),isocalc_sig=isocalc_sig,isocalc_resolution=isocalc_resolution)
        pickle.dump(mz_list_tmp,open(load_file,'w'))

    for sum_formula in mz_list_tmp:
            if not sum_formula in mz_list:
                mz_list[sum_formula] = {}
            mz_list[sum_formula][adduct] = mz_list_tmp[sum_formula][adduct]

print 'all isotope patterns generated and loaded'

# Parse data
from pyIMS.hdf5.inMemoryIMS_hdf5 import inMemoryIMS_hdf5
from pyIMS.image_measures import level_sets_measure
IMS_dataset=inMemoryIMS_hdf5(filename_in)
#IMS_dataset=inMemoryIMS_hdf5(filename_in,max_mz=500.,min_int=150.,index_range=range(0,3842)) #options to limit data size

## The main part of the algorithm - don't change these parameters
tstart=time.time()

measure_value_score={}
iso_correlation_score = {}
iso_ratio_score = {}
for n, sum_formula in enumerate(sum_formulae):
    for adduct in adducts:
        ## 1. Geneate ion images
        mz_list[sum_formula][adduct][0] #get centroid mz values
        ion_datacube = IMS_dataset.get_ion_image(mz_list[sum_formula][adduct][0],ppm) #for each spectrum, sum the intensity of all peaks within tol of mz_list
        # tidy images
        for xic in ion_datacube.xic:
            xic_q = np.percentile(xic,q)
            xic[xic>xic_q]=xic_q
        ## 2. Spatial Chaos 
        if not sum_formula in measure_value_score:
            measure_value_score[sum_formula] = {}

        if np.sum(ion_datacube.xic_to_image(0)) == 0:
            measure_value_score[sum_formula][adduct] = 0 # this is now coded into measure_of_chaos
        else:
            measure_value_score[sum_formula][adduct] = 1-level_sets_measure.measure_of_chaos(ion_datacube.xic_to_image(0),nlevels,interp=False)[0]
            if measure_value_score[sum_formula][adduct] == 1:
                 measure_value_score[sum_formula][adduct] = 0
            clear_output(wait=True)
            
        ## 3. Score correlation with monoiso
        if not sum_formula in iso_correlation_score:
            iso_correlation_score[sum_formula] = {}
        if len(mz_list[sum_formula][adduct][1]) > 1:
            iso_correlation = np.corrcoef(ion_datacube.xic)[1:,0]
            iso_correlation[np.isnan(iso_correlation)] = 0 # when alll values are the same (e.g. zeros) then correlation is undefined
            iso_correlation_score[sum_formula][adduct] = np.average(
                    iso_correlation,weights=mz_list[sum_formula][adduct][1][1:]
                     ) # slightly faster to compute all correlations and pull the elements needed
        else: # only one isotope peak, so correlation doesn't make sense
            iso_correlation_score[sum_formula][adduct] = 1
    
        ## 4. Score isotope ratio
        if not sum_formula in iso_ratio_score:
            iso_ratio_score[sum_formula] = {}
        isotope_intensity = mz_list[sum_formula][adduct][1] 
        image_intensities = [sum(ion_datacube.xic[ii]) for ii in range(0,len(mz_list[sum_formula][adduct][0]))]
        iso_ratio_score[sum_formula][adduct] = 1-np.mean(abs( isotope_intensity/np.linalg.norm(isotope_intensity) - image_intensities/np.linalg.norm(image_intensities)))    
    if np.mod(n,10)==0:
            clear_output(wait=True)
            print '{:3.2f}\% complete\r'.format(100*n/len(sum_formulae),end="\r")
            sys.stdout.flush()
print 'Elapsed: {:5.2f} seconds'.format(time.time() - tstart)
        

# Save the processing results
filename_out = '{}{}{}_full_results.txt'.format(output_dir,os.sep,os.path.splitext(os.path.basename(filename_in))[0])
with open(filename_out,'w') as f_out:
    f_out.write('sf,adduct,mz,moc,spec,spat,pass\n'.format())
    for sum_formula in sum_formulae:
        for adduct in adducts:
            moc_pass =  measure_value_score[sum_formula][adduct] > measure_tol and iso_correlation_score[sum_formula][adduct] > iso_corr_tol and iso_ratio_score[sum_formula][adduct] > iso_ratio_tol
            f_out.write('{},{},{},{},{},{},{}\n'.format(
                    sum_formula,
                    adduct,
                    mz_list[sum_formula][adduct][0][0],
                    measure_value_score[sum_formula][adduct],
                    iso_correlation_score[sum_formula][adduct],
                    iso_ratio_score[sum_formula][adduct],
                    moc_pass)) 

filename_out = '{}{}{}_pass_results.txt'.format(output_dir,os.sep,os.path.splitext(os.path.basename(filename_in))[0])
with open(filename_out,'w') as f_out:
    f_out.write('ID,sf,adduct,mz,moc,spec,spat\n'.format())
    for sum_formula in sum_formulae:
        for adduct in adducts:
            if measure_value_score[sum_formula][adduct] > measure_tol and iso_correlation_score[sum_formula][adduct] > iso_corr_tol and iso_ratio_score[sum_formula][adduct] > iso_ratio_tol:
                f_out.write('{},{},{},{},{},{},{}\n'.format(
                    sum_formulae[sum_formula]['db_id'],
                    sum_formula,adduct,
                    mz_list[sum_formula][adduct][0][0],
                    measure_value_score[sum_formula][adduct],
                    iso_correlation_score[sum_formula][adduct],
                    iso_ratio_score[sum_formula][adduct]))
                       

## Re-Load results
#from pySpatialMetabolomics.tools import results_tools
#filename_out = '{}{}{}_full_results.txt'.format(output_dir,os.sep,os.path.splitext(os.path.basename(filename_in))[0])
#(measure_value_score,iso_correlation_score,iso_ratio_score,moc_pass) = results_tools.load_results(filename_out)

## Make Parula colormap
# todo - generate unique colourmap to avoid IP issues 
from pySpatialMetabolomics.tools import colourmaps
c_map = colourmaps.make_cmap([(53,42,135)
,(2,104,225)
,(16,142,210)
,(15,174,185)
,(101,190,134)
,(192,188,96)
,(255,195,55)
,(249,251,14)],bit=True,name='parula')

## Output pass molecules
def check_pass(pass_thresh,pass_val):
    tf = []
    for v,t in zip(pass_val,pass_thresh):
        tf.append(v>t)
    if all(tf):
        return True
    else:
        return False

def plot_images(ion_datacube,iso_spect,iso_max):
    fig = plt.figure(figsize=(20,15),dpi=300)
    ax = [   plt.subplot2grid((2, 4), (0, 0)),
         plt.subplot2grid((2, 4), (0, 1)),
         plt.subplot2grid((2, 4), (0, 2)),
         plt.subplot2grid((2, 4), (0, 3)),
         plt.subplot2grid((2, 4), (1, 0), colspan=4, rowspan=1)
     ]
    for a in ax:
        a.cla()
    # plot images
    for ii in range(0,iso_max):
        im = ion_datacube.xic_to_image(ii)
        # hot-spot removal
        notnull=im>0 
        if np.sum(notnull==False)==np.shape(im)[0]*np.shape(im)[1]:
            im=im
        else:
            im_q = np.percentile(im[notnull],q_val)
            im_rep =  im>im_q       
            im[im_rep] = im_q 

        ax[ii].imshow(im,cmap=c_map)
        ax[ii].set_title('m/z: {:3.4f}'.format(mz_list[sum_formula][adduct][0][ii]))
    # plot spectrum
    notnull=ion_datacube.xic_to_image(0)>0
    data_spect = [np.sum(ion_datacube.xic_to_image(ii)) for ii in range(0,iso_max)]
    data_spect = data_spect / np.linalg.norm(data_spect)
    iso_spect = iso_spect/np.linalg.norm(iso_spect)

    markerline, stemlines, baseline = ax[4].stem( mz_list[sum_formula][adduct][0][0:iso_max],iso_spect,'g')
    plt.setp(stemlines, linewidth=2, color='g')     # set stems  colors
    plt.setp(markerline, 'markerfacecolor', 'g','markeredgecolor','g')    # make points 

    markerline, stemlines, baseline = ax[4].stem( mz_list[sum_formula][adduct][0][0:iso_max],data_spect,'r')
    plt.setp(stemlines, linewidth=2, color='r')     # set stems colors
    plt.setp(markerline, 'markerfacecolor', 'r','markeredgecolor','r')    # make points 
    
    #plot proxy artist
    proxies=[]
    h, = plt.plot(mz_list[sum_formula][adduct][0][0],[0],'-g')
    proxies.append(h)
    h, = plt.plot(mz_list[sum_formula][adduct][0][0],[0],'-r')
    proxies.append(h)

    
    ax[4].legend(proxies,('predicted pattern','data pattern'), numpoints=1)
    return fig,ax

pass_formula = []
n_iso=4
q_val=99

for sum_formula in mz_list:
    for adduct in adducts:
        ch_pass = check_pass((measure_tol,iso_corr_tol,iso_ratio_tol),(measure_value_score[sum_formula][adduct],iso_correlation_score[sum_formula][adduct],iso_ratio_score[sum_formula][adduct])) 
        if ch_pass:
            pass_formula.append('{} {}'.format(sum_formula,adduct))
            ion_datacube = IMS_dataset.get_ion_image(mz_list[sum_formula][adduct][0],ppm)
            iso_max = min((n_iso,len(mz_list[sum_formula][adduct][0])))
            iso_spect = mz_list[sum_formula][adduct][1][0:iso_max]
            fig, ax = plot_images(ion_datacube,iso_spect,iso_max)
            fig.suptitle('{} [M+{}]. Measure of Chaos: {:3.4f}, Image Correlation: {:3.4f} Isotope Score: {:3.4f}'.format(sum_formula,adduct,measure_value_score[sum_formula][adduct],iso_correlation_score[sum_formula][adduct],iso_ratio_score[sum_formula][adduct]))
            #SAVE figure
            plt.savefig('{}/{}_{}_{}.png'.format(output_dir,os.path.basename(IMS_dataset.file_dir),sum_formula,adduct))
            plt.clf()

# View the output from a specific metabolite
sum_formula='C21H21N'
adduct = 'H'
q=99
iso_correlation = np.corrcoef(ion_datacube.xic)[1:,0]
iso_correlation[np.isnan(iso_correlation)] = 0 # when alll values are the same (e.g. zeros) then correlation is undefined



ion_datacube = IMS_dataset.get_ion_image(mz_list[sum_formula][adduct][0],ppm)
iso_max = min((n_iso,len(mz_list[sum_formula][adduct][0])))
iso_spect = mz_list[sum_formula][adduct][1][0:iso_max]
fig, ax = plot_images(ax,ion_datacube,iso_spect,iso_max)
fig.suptitle('{} [M+{}]. Measure of Chaos: {:3.4f}, Image Correlation: {:3.4f} Isotope Score: {:3.4f}'.format(sum_formula,adduct,measure_value_score[sum_formula][adduct],iso_correlation_score[sum_formula][adduct],iso_ratio_score[sum_formula][adduct]))



