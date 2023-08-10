import pandas as pd

import sqlite3
con = sqlite3.connect("nhsadmin.sqlite")

gp = pd.read_sql_query('SELECT * from epraccur', con)
gp.head()

dentists=pd.read_sql_query('SELECT * from egdpprac', con)
dentists.head()

#Create a function to grab a zip file from an online location and then grab a specified file from inside it
import requests, zipfile

try:
    from StringIO import StringIO as zreader
except ImportError:
    from io import BytesIO as zreader
    
def zipgrabber(url):
    r = requests.get(url)
    z = zipfile.ZipFile(zreader(r.content))
    #List contents with:
    #z.namelist()
    return z

def csvfileopener(z,k,metadata=False):
    f=None
    for n in z.namelist():
        if n.split('.')[-1]=='csv' and k in n: f=n
    if f is not None:
        df = pd.read_csv(z.open(f))
        df.dropna(how='all',axis=1,inplace=True)
        df=df[[c for c in df.columns if not c.startswith('Unnamed')]]
        if metadata:
            complaints_meta = get_metadata(z)
            cols=['::'.join(x) for x in zip(complaints_meta['Name'],complaints_meta['meta'])]
            df.columns=cols
        return df
    return pd.DataFrame()

def get_metadata(z):
    for n in z.namelist():
        if 'xls' in n and 'meta' in n: f=n
    xl=pd.ExcelFile(z.open(f))
    for n in xl.sheet_names:
        if 'KO41b' in n: sheetname=n
    complaints_meta=xl.parse(sheetname,header=None)
    complaints_meta.columns=['Name','typ','meta']
    complaints_meta = complaints_meta[complaints_meta['Name']!='Column']
    return complaints_meta

url='http://www.digital.nhs.uk/catalogue/PUB21533/data-writ-comp-nhs-2015-2016-csv.zip'
zip_2015_16=zipgrabber(url)

complaints_2015_16=csvfileopener(zip_2015_16,'KO41a')
complaints_2015_16.head(3)

complaints_b_2015_16=csvfileopener(zip_2015_16,'KO41b',True)
complaints_b_2015_16.head(3)

#join complaints parent codes
parentGP=gp[['Organisation Code','Commissioner']]
parentGP.columns=['Organisation Code','Parent Organisation Code']
parentDentist=dentists[['Organisation Code','Parent Organisation Code']]
codeparent=pd.concat([parentGP,parentDentist])
codeparent.head(3)

iwcodes=codeparent[(codeparent['Parent Organisation Code']=='10L')]
complaints_iw=pd.merge(complaints_b,iwcodes, left_on='Practice_Code::Dental/GP Practice Code', right_on='Organisation Code')
gp[gp['Organisation Code'].isin(complaints_iw['Practice_Code::Dental/GP Practice Code'].tolist())].head()

import re

def epracurrDetails(code):
    return gp[gp['Organisation Code']==code].to_dict(orient='records')[0]

def getAddress(d):
    c=[x for x in d.keys() if 'address line' in x.lower()]
    c.append('Postcode')
    return [d[x] for x in c if ((d[x] is not None) and not pd.isnull(d[x]))]  

def receivedUpheld(d,k):
    rxd=d[k]['received'] if 'received' in d[k] else 0
    upheld=d[k]['upheld'] if 'upheld' in d[k] else 0
    return rxd,upheld

def receivedUpheldArea(d,k,hack2014=False):
    if hack2014: return receivedUpheld(d,k)
    rxd=d[k]['Complaints by area'] if 'Complaints by area' in d[k] else 0
    upheld=d[k]['Complaints upheld by area'] if 'Complaints upheld by area' in d[k] else 0
    return rxd,upheld

def receivedUpheldSubject(d,k,hack2014=False):
    if hack2014: return receivedUpheld(d,k)
    rxd=d[k]['Complaints by subject'] if 'Complaints by subject' in d[k] else 0
    upheld=d[k]['Complaints upheld by subject'] if 'Complaints upheld by subject' in d[k] else 0
    return rxd,upheld

def complaintReport(item,hack2014=False):
    details=epracurrDetails(item['Practice_Code::Dental/GP Practice Code'])
    txt='\n---------------\n'
    txt=txt+'Complaint report for: {name} ({code})\n'.format(name=details['Name'],
                                                      code=item['Practice_Code::Dental/GP Practice Code'])
    txt=txt+'\nAddress: {addr}.'.format(addr=', '.join(getAddress(details)))
    if ((details['Contact Telephone Number'] is not None) and not pd.isnull(details['Contact Telephone Number'])):
        txt=txt+'\nTelephone: {tel}'.format(tel=details['Contact Telephone Number'])
    txt=txt+'\n'
    
    if hack2014:
        complaints_received={'Complaint by subject':{},'Complaint by area':{}}
        complaints_upheld={'Complaint by subject':{},'Complaint by area':{}}
    else:
        complaints_received={'Complaints by subject':{},'Complaints by area':{}}
        complaints_upheld={'Complaints upheld by subject':{},'Complaints upheld by area':{}}
    
    complaints_by_area={}
    complaints_by_subject={}
    for col in item.index.values:
        if 'complaint' in col.lower() and item[col]>0:
            
            if hack2014:
                k=col.replace('\n','').split('Total number of written complaints ')

                if k[0]=='': k[0]='Total'
                k[0]=k[0].strip()
                k[1]=k[1]

                if 'received' in k[1]:
                    complaints_received[k[1].split('::')[1]][k[0]]=int(item[col])   
                elif 'upheld' in k[1]:
                    complaints_upheld[k[1].split('::')[1]][k[0]]=int(item[col])

                if 'area' in k[1]:
                    if k[0] not in complaints_by_area: complaints_by_area[k[0]]={}
                    complaints_by_area[k[0]][k[1].split('::')[0]]=int(item[col])
                elif 'subject' in k[1]:
                    if k[0] not in complaints_by_subject: complaints_by_subject[k[0]]={}
                    complaints_by_subject[k[0]][k[1].split('::')[0]]=int(item[col])
            else:
                if 'TOTAL' in col:
                    k=re.compile(r' SUBJECT TOTAL| TOTAL').split(col)
                    k[1]=k[1].lstrip('::')
                    complaints_received[k[1]][k[0]]=int(item[col])   
                elif 'UPHELD' in col:
                    k=re.compile(r' SUBJECT UPHELD| UPHELD').split(col)
                    k[1]=k[1].lstrip('::')
                    complaints_upheld[k[1]][k[0]]=int(item[col])

                if 'area' in col:
                    if k[0] not in complaints_by_area: complaints_by_area[k[0]]={}
                    complaints_by_area[k[0]][k[1].split('::')[0]]=int(item[col])
                elif 'subject' in col:
                    if k[0] not in complaints_by_subject: complaints_by_subject[k[0]]={}
                    complaints_by_subject[k[0]][k[1].split('::')[0]]=int(item[col])
                
    txt=txt+'\n'

    if complaints_by_area!={}:
        rxd,upheld=receivedUpheldArea(complaints_by_area,"Total",hack2014)
        txt=txt+'Complaints by area ({} received, of which {} upheld):'.format(rxd,upheld)
        for complaint in complaints_by_area:
            if complaint=='Total': continue
            rxd,upheld=receivedUpheldArea(complaints_by_area,complaint,hack2014)
            txt=txt+'\n    - {}: {} received, of which {} upheld.'.format(complaint,rxd,upheld)
        txt=txt+'\n\n'
    else: txt=txt+'No complaints by area.\n' 
        
    if complaints_by_subject!={}:
        rxd,upheld=receivedUpheldSubject(complaints_by_subject,"Total",hack2014)
        txt=txt+'Complaints by subject ({} received, of which {} upheld):'.format(rxd,upheld)
        for complaint in complaints_by_subject:
            if complaint=='Total': continue
            rxd,upheld=receivedUpheldSubject(complaints_by_subject,complaint,hack2014)
            txt=txt+'\n    - {}: {} received, of which {} upheld.'.format(complaint,rxd,upheld)
    else: txt=txt+'No complaints by subject.\n\n'      
    #print(txt)
    #print('\n---------------\n')
    
    return txt

reports=complaints_iw.apply(lambda x: complaintReport(x), axis=1 )

print('''
National Statistics Data on Written Complaints in the NHS - 2015-16 [NS]
Publication date: September 15, 2016
http://www.digital.nhs.uk/catalogue/PUB21533
''')

for report in reports:
    print(report)

url='http://www.hscic.gov.uk/catalogue/PUB18021/data-writ-comp-nhs-2014-2015-csv.zip'
zip_2014_15=zipgrabber(url)
complaints_2014_15=pd.read_csv(zip_2014_15.open('Data on Written Complaints in the NHS 2014-15 KO41a csv.csv'))
complaints_2014_15.dropna(how='all',axis=1,inplace=True)
complaints_2014_15.head(3)

get_metadata(zip_2014_15)

complaints_b_2014_15=csvfileopener(zip_2014_15,'KO41b',True)
complaints_b_2014_15.head(3)

#join complaints parent codes
parentGP=gp[['Organisation Code','Commissioner']]
parentGP.columns=['Organisation Code','Parent Organisation Code']
parentDentist=dentists[['Organisation Code','Parent Organisation Code']]
codeparent=pd.concat([parentGP,parentDentist])
codeparent.head(3)

iwcodes=codeparent[(codeparent['Parent Organisation Code']=='10L')]
complaints_iw=pd.merge(complaints_b_2014_15,iwcodes, left_on='Practice_Code::Dental/GP Practice Code', right_on='Organisation Code')
gp[gp['Organisation Code'].isin(complaints_iw['Practice_Code::Dental/GP Practice Code'].tolist())].head()

reports=complaints_iw.apply(lambda x: complaintReport(x,hack2014=True), axis=1 )

print('''
National Statistics Data on Written Complaints in the NHS - 2014-15 [NS]
Publication date: ??, 2015
http://www.digital.nhs.uk/catalogue/PUB21533
''')

for report in reports:
    print(report)



