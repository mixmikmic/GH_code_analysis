get_ipython().magic('pylab inline')
import seaborn as sns
from astropy.io import ascii
from astropy.table import Table
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.stats import mad_std
import pandas

# Read the 3D-HST master catalog.
download_file("http://monoceros.astro.yale.edu/RELEASE_V4.1.5/3dhst.v4.1.5.master.fits.gz")
mastercat_file = "3dhst.v4.1.5.master.fits.gz"
pmcat = Table.read(mastercat_file).to_pandas()
# Note, an endian-ness mismatch between FITS and numpy can cause
# gruesome errrors if you import this WRONG way:
#   (mcat, mcat_hdr) = fits.getdata(mastercat_file, header=True)
#   pmcat = pandas.DataFrame.from_records(mcat)
# USE .to_pandas() to avoid this.  See https://github.com/astropy/astropy/issues/1156

print type(pmcat)
pandas.set_option('display.max_columns', 500)  # print all columns in .head()
pmcat.shape

# ipython notebook has nice formatting of pandas.head()
pmcat['JRRID'] =  pmcat['field'] + "." + pmcat['phot_id'].astype(str)
pmcat.head()

# Demo some basic filtering.  Not used below.
zlo = 2.0
zhi = 3.0
Mlo = 9.0
Mhi = 9.5
print pmcat[pmcat['use_phot'].eq(1)].shape   # filter on good photometry
print pmcat[pmcat['z_best'].between(zlo,zhi)].shape  # Filter redshift
print pmcat[pmcat['lmass'].between(Mlo,Mhi)].shape   # Filter stellar mass
filt = pmcat['z_best'].between(zlo,zhi) & pmcat['lmass'].between(Mlo,Mhi) & pmcat['use_phot'].eq(1)
print pmcat[filt].shape #all 3 filters

# Read in Arjen's catalogs of morphological parameters, for each field
#download_file("http://www.mpia-hd.mpg.de/homes/vdwel/allfields.tar")
# mkdir VanderWel; cd VanderWel; tar -xvf allfields.tar
# I then edited by hand the headers, stripping out the initial "#  "
vdw_H_files = ("VanderWel/aegis/aegis_3dhst.v4.1_f125w.galfit", "VanderWel/goodss/goodss_3dhst.v4.1_f125w.galfit", "VanderWel/cosmos/cosmos_3dhst.v4.1_f125w.galfit", "VanderWel/uds/uds_3dhst.v4.1_f125w.galfit", "VanderWel/goodsn/goodsn_3dhst.v4.1_f125w.galfit")
vdw_fields = ('aegis', 'goodss', 'cosmos', 'uds', 'goodsn')
vdw_df = {}
for ii, label in enumerate(vdw_fields):
    vdw_df[label] = pandas.read_table(vdw_H_files[ii], delim_whitespace=True, comment="#", header=0)
    vdw_df[label]['field'] = vdw_fields[ii]
    vdw_df[label]['JRRID'] = vdw_fields[ii] + '.' + vdw_df[label]['NUMBER'].astype(str)
vdw_all = pandas.concat(vdw_df)

vdw_all.tail()

pmcat_jrr   = pmcat.set_index('JRRID')
vdw_all_jrr= vdw_all.set_index('JRRID')

# Filter on redshift and M*
zlo = 2.0
zhi = 3.0
Mlo = 9.0
Mhi = 9.5
# This filtering works b/c I re-indexed arrays by unique index 'JRRID'. 
# If I weren't already a fan of pandas, this would sell me.  filt1 and filt2 are filters
# created from the first and second catalogs, respectively, and while they have different
# shapes, Pandas can do math on them (in this case, boolean and), because their indexes
# are in common.  This makes a difficult matching (catalogs of different sizes) easy.
filt1 = pmcat_jrr['z_best'].between(zlo,zhi) & pmcat_jrr['lmass'].between(Mlo,Mhi)
print pmcat_jrr[filt1].shape #both
print vdw_all_jrr[filt1].shape
filt2 = vdw_all_jrr['f'].eq(0)   # Filter on a good Galfit fit
filt = filt1 & filt2   # Require galaxy in M*/z selection, and a good Galfit
print vdw_all_jrr[filt2].shape, "N with good galfit"
print vdw_all_jrr[filt].shape, "N with good M*, z, galfit:"

# Add a point I want to plot
kpc_per_arcsec = 8.085
# add in s1110          r_e(")          n     d(r_e("))        d(n)
s1110_candelized= (2.3/kpc_per_arcsec, 1.0, 0.3/kpc_per_arcsec, 0.4)   # average over all bands
s1110_F160cand = (0.32, 0.68, 0.029, 0.26)   #F160W only, from Table 1 of paper draft
def plot_my_point() :
    plt.scatter( s1110_candelized[0], s1110_candelized[1], color='k', s=100)
    plt.errorbar(s1110_candelized[0],s1110_candelized[1], xerr=s1110_candelized[2],yerr=s1110_candelized[3], ecolor='k')
    plt.scatter( s1110_F160cand[0], s1110_F160cand[1], color='r', s=100)
    plt.errorbar(s1110_F160cand[0],s1110_F160cand[1], xerr=s1110_F160cand[2],yerr=s1110_F160cand[3], ecolor='r')

# Let's try the seaborn JointGrid, to add histograms to margins
# code borrowed from https://github.com/mwaskom/seaborn/issues/469
sns.set(font_scale=2)
sns.set_style("white")
plotted = vdw_all_jrr[filt]
print "median, mad:", plotted.re.median()*kpc_per_arcsec, mad_std(plotted.re)*kpc_per_arcsec
print "s1110 R_e, std:", s1110_F160cand[0]*kpc_per_arcsec, s1110_F160cand[2]*kpc_per_arcsec,
g = sns.JointGrid(plotted.re, plotted.n, size=8)
g.set_axis_labels("effective radius (\")", "Sersic index")
g.ax_marg_x.hist(plotted.re, bins=np.arange(0, 1.0, 0.05))
g.ax_marg_y.hist(plotted.n, bins=np.arange(0, 5, 0.2), orientation="horizontal")
g.plot_joint(plt.hexbin, gridsize=40, extent=[0, 1.4, 0, 5], cmap="Blues")
plot_my_point()  # cool, I can overplot regular matplotlib commands like scatter, errorbar
g.fig.savefig("filter_hex.pdf")



