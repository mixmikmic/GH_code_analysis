get_ipython().magic('pylab inline')
from astropy.io import ascii
from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns

# This big file takes some time to download.  May time out. If so, wget, direct link, move on
file = "http://iopscience.iop.org/0067-0049/203/2/24/suppdata/apjs449744t2_mrt.txt"
#file_local = "/Volumes/Apps_and_Docs/jrrigby1/Dropbox/SGAS-shared/s1110-paper2/JRR_morph_working/VanderWel/apjs449744t2_mrt.txt"
temp_table = ascii.read(file)

print "Have read in Table 2 of van der Wel 2012 as an Astropy table"
table = temp_table.to_pandas()  # Convert from astropy Table to Pandas Data Frame.  Needs astropy 1.2
print "Have converted to a Pandas Data frame"
table.head()

good = table[table['Flag'].eq(0)]  # filter out bad Galfit fits
H = good[good['Filter'].str.contains("H")]  # filter just H-band
J = good[good['Filter'].str.contains("J")]  # filter just J-band
small = H[H['ID'].lt(3000)]  # make a small subset of H, for quick plotting

# Add a point I want to plot
kpc_per_arcsec = 8.085
# add in s1110            _e(")          n     d(r_e("))        d(n)
s1110_candelized= (2.3/kpc_per_arcsec, 1.0, 0.3/kpc_per_arcsec, 0.4)   # average over all bands
s1110_F160cand = (0.32, 0.68, 0.029, 0.26)   #F160W only, from Table 1 of paper draft
def plot_my_point() :
    plt.scatter( s1110_candelized[0], s1110_candelized[1], color='k', s=100)
    plt.errorbar(s1110_candelized[0],s1110_candelized[1], xerr=s1110_candelized[2],yerr=s1110_candelized[3], ecolor='k')
    plt.scatter( s1110_F160cand[0], s1110_F160cand[1], color='r', s=100)
    plt.errorbar(s1110_F160cand[0],s1110_F160cand[1], xerr=s1110_F160cand[2],yerr=s1110_F160cand[3], ecolor='r')

# plot the result as a density plot.  
plotted = H   #small to run fast, H to run slow but more accurate
sns.set(font_scale=2)
sns.set_style("white")
ax = sns.kdeplot(plotted.r, plotted.n, n_levels=22)
ax.set_xlim(0,1.5)
ax.set_ylim(0,5)
ax.set_xlabel("effective radius (\")")
ax.set_ylabel("Sersic index")
plot_my_point()  # cool, I can overplot regular matplotlib commands like scatter, errorbar
#savefig("~/Dropbox/SGAS-shared/s1110-paper2/JRR_morph_working/VanderWel/kde.pdf")
savefig("Output/kde.pdf")

# Let's try the seaborn JointGrid, to add histograms to margins
# code borrowed from https://github.com/mwaskom/seaborn/issues/469
plotted = H  #small to run fast, H to run slow but more accurate
g = sns.JointGrid(plotted.r, plotted.n, size=8)
g.set_axis_labels("effective radius (\")", "Sersic index")
g.ax_marg_x.hist(plotted.r, bins=np.arange(0, 1.4, 0.05))
g.ax_marg_y.hist(plotted.n, bins=np.arange(0, 5, 0.2), orientation="horizontal")
g.plot_joint(plt.hexbin, gridsize=40, extent=[0, 1.4, 0, 5], cmap="Blues")
plot_my_point()  # cool, I can overplot regular matplotlib commands like scatter, errorbar
#g.fig.savefig("~/Dropbox/SGAS-shared/s1110-paper2/JRR_morph_working/VanderWel/hex.pdf")
g.fig.savefig("Output/hex.pdf")

