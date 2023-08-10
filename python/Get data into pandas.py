import pandas
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.data import download_file
import subprocess

# Read a simple ascii text file into Pandas
get_ipython().system('head -n 5 Sample_data/zody_and_ism.txt')
df1 = pandas.read_table("Sample_data/zody_and_ism.txt", delim_whitespace=True, comment="#")
df1.head(2)

myfile = "Sample_data/weirdheader.txt"
print subprocess.check_output("head -3 " + myfile, shell=True)
subprocess.check_output("sed s/\#// < " + myfile + "> /tmp/workaround", shell=True)
#!sed 's/#//' < Sample_data/weirdheader.txt > /tmp/workaround
df = pandas.read_table("/tmp/workaround", sep='\s+', skipinitialspace=True)
# Jane has coded this into jrr.util.strip_pound_before_colnames():
df.head(2)

# Read a simple .csv (comma-separated variable) file into Pandas
df2 = pandas.read_csv("Sample_data/thermal_curve_jwst_jrigby_1.1.csv", comment="#", names=("wave", "bkg"))
df2.head()

# Read a machine-readable table from an ApJ paper into Pandas, via astropy.Table
file2 = "http://iopscience.iop.org/2041-8205/814/1/L6/suppdata/apjl521409t1_mrt.txt"
temp_table = ascii.read(file2) # this automagically gets the format right.
df3 = temp_table.to_pandas()  # Convert from astropy Table to Pandas Data Frame.  Needs astropy 1.2
df3.head(1)

# Read a binary .fits table into Pandas, via astropy.Table
stsci_file = "Sample_data/example_bkgs.fits"
tab = Table.read(stsci_file)
stsci_df = tab.to_pandas()
# If you're feeling fancy, you can do it in one line:
stsci_df1 = Table.read(stsci_file).to_pandas()
stsci_df1.tail()

# Read a binary .fits table with a really gnarly format:  7,000 x3 columns wide, 1 line long
zhu_file = 'https://data.sdss.org/sas/dr13/eboss/elg/composite/v1_0/eBOSS_ELG_NUV_composite.fits'
#mytab = Table.read(zhu_file)
#df = mytab.to_pandas()  # This fails b/c the file has a ridiculous format.  7,000 x3 columns wide, 1 line long
t = fits.open(zhu_file)  # What follows is a tedious workaround.
tbdata = t[1].data
print tbdata.columns
wave = tbdata['WAVE'].T
df = pandas.DataFrame(data=wave, columns=('wave',))
df['fluxmedian']     = tbdata['FLUXMEDIAN'].T
df['fluxmedian_err'] = tbdata['FLUXMEDIAN_ERR'].T
df.head()

# Note, an endian-ness mismatch between FITS and numpy can cause
# gruesome errrors if you import this WRONG way:
#   (mcat, mcat_hdr) = fits.getdata(mastercat_file, header=True) #WRONG
#   pmcat = pandas.DataFrame.from_records(mcat)  # WRONG
# USE .to_pandas() to avoid this.  See https://github.com/astropy/astropy/issues/1156

# You can read really big binary fits tables.  May take a while to download.
# Here, let's read the 3D-HST master catalog.
download_file("http://monoceros.astro.yale.edu/RELEASE_V4.1.5/3dhst.v4.1.5.master.fits.gz")
#mastercat_file = "3dhst.v4.1.5.master.fits.gz"
#pmcat = Table.read(mastercat_file).to_pandas()

