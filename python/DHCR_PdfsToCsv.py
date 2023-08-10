#download pdfs from website

import urllib2

borolist= ['manhattan','bronx','brooklyn','queens','statenisl']

for boro in borolist:
    url='http://www1.nyc.gov/assets/rentguidelinesboard/pdf/2016'+boro+'bldgs.pdf'
    filedata = urllib2.urlopen(url)  
    datatowrite = filedata.read()
    
    with open('dhcr_'+boro+'.pdf', 'wb') as f:  
        f.write(datatowrite)

# scrape pdf tables into a dataframe
import pandas as pd
from tabula import read_pdf
import pyPdf

dhcr = pd.DataFrame()
for boro in borolist:
    filename= 'dhcr_'+boro+'.pdf'
    n=pyPdf.PdfFileReader(open(filename)).getNumPages()
    df=read_pdf(filename,pages=range(1,n+1))
    df['Borough']=borolist.index(boro)+1
    dhcr=pd.concat([dhcr,df], ignore_index=True)
    
dhcr['Borough'] = pd.to_numeric(dhcr.Borough,errors='coerse')
dhcr['BLOCK'] = pd.to_numeric(dhcr.BLOCK,errors='coerse')
dhcr['LOT'] = pd.to_numeric(dhcr.LOT,errors='coerse')

dhcr['BBL'] = dhcr.Borough.apply(lambda x: '{:01.0f}'.format(x)).astype(str) +dhcr.BLOCK.apply(lambda x: '{:05.0f}'.format(x)).astype(str) +dhcr.LOT.apply(lambda x: '{:04.0f}'.format(x)).astype(str)

dhcr['BBL'] = pd.to_numeric(dhcr.BBL,errors='coerse')

dhcr.shape

dhcr.head()

pd.DataFrame({'unique values':dhcr.apply(lambda x: x.nunique()),
                         'count':dhcr.apply(lambda x: x.count())})

# write to csv file

dhcr.to_csv('dhcr.csv',index=False)



