# Adjust this path to have the relative (or absolute) path to the processed data.
datapath="../data/Sol_16208/20201001001"

import os
from astropy.io import fits

for file in os.listdir(datapath+"/event_cl/"):
    if file.find("06") == -1 or file.find("chu") == -1:
        continue
    if not file.endswith(".evt"):
        continue
        
    evtfile=datapath+'/event_cl/'+file
    print(file)
    hdulist = fits.open(evtfile)
    print()
    print('Exposure: ',hdulist[1].header['EXPOSURE'])
    print('Number of counts: ', hdulist[1].header['NAXIS2'])
    print()
    hdulist.close()





