fname = ("/Users/klay6683/Dropbox/SternchenAndMe/UVIS_Enc_Occ_2016_03_11"
         "/HSP2016_03_11_11_48_26_000_UVIS_233EN_ICYEXO001_PIE")

from pyuvis.io import HSP

hsp = HSP(fname, freq='2ms')

hsp.series.resample('1s').sum().head()

hsp.counts_per_sec.resample('1s').mean().head()

hsp.counts_per_sec.head()

