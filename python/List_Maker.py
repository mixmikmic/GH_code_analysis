import numpy as np
import apogee.tools.read as apread
from matplotlib import pyplot as plt
from astropy.io import fits
import random
import csv

with open('All Visits4.csv') as f:
    x = list(csv.reader(f,delimiter = '\t'))
    count = len(x)
    with open('Non_binary.csv','w') as csvfile:
        names = ['Location ID','Apogee ID', 'Binary']
        writer = csv.DictWriter(csvfile,delimiter='\t',fieldnames=names)
        writer.writeheader()
        for i in range(1100):
            y = random.randint(0,count)
            writer.writerow({'Location ID':x[y][0],'Apogee ID': x[y][1], 'Binary': 0})
            

LocID = []
ApogID = []
with open('Non_binary.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    next(reader,None)
    
    for row in reader:
        b = int(row[0])
        c = (row[1])
        LocID.append(b)
        ApogID.append(c)

for i in range(len(LocID)):
    locationID = LocID[i]
    apogeeID = ApogID[i]
    
    header = apread.apStar(locationID, apogeeID, ext=0, header=True)
    Data = apread.apStar(locationID, apogeeID, ext=9, header=False)
    
    nvisits = header[1]['NVISITS']
    plt.figure(figsize=(10,10))
    for visit in range(0, nvisits):
        if (nvisits != 1):
            CCF = Data['CCF'][0][2+ visit]
        else:
            CCF = Data['CCF'][0]

        plt.plot(CCF + visit,label= 'Visit: '+str(1+visit))
        plt.xlabel('CCF Lag',fontsize=15)
        plt.ylabel('$\widehat{CCF}$ Units', fontsize=15)
        plt.title(' All Visits for'+ str(apogeeID),fontsize=16)
        pl.legend(loc='lower left')
        plt.savefig('NBS_'+str(locationID)+'_'+str(apogeeID)+'.png',dpi=900)
    plt.close('all')

locID = []
apoID = []
binary = []
with open('Non_binary.csv','r') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    next(reader,None)
    for row in reader:
        a = int(row[0][0:4])
        b = (row[0][5:23])
        c = int(row[0][24])
        locID.append(a)
        apoID.append(b)
        binary.append(c)

with open('Non_Binaries.csv','w') as files:
           headers = ['Location ID', 'Apogee ID', 'Binary']
           writer = csv.DictWriter(files,delimiter = '\t', fieldnames=headers)
           writer.writeheader()
           for i in range(1250):
               writer.writerow({'Location ID':locID[i],'Apogee ID': apoID[i], 'Binary': binary[i]})
   



