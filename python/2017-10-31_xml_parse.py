#this works really well if not series dependent, but won't work for all
from urllib.request import urlretrieve
import os
import xml.etree.ElementTree as ET
import pandas as pd

PATH = '../../output/chip/xmls/'
gse = 'GSE49102'
xml = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+gse+'&targ=gsm&view=full&form=xml'
os.makedirs(PATH, exist_ok=True)

urlretrieve(xml, PATH+gse+'.xml')
tree = ET.parse('../../output/chip/GSE49102.xml')
root = tree.getroot()
ns = {'url': 'http://www.ncbi.nlm.nih.gov/geo/info/MINiML'}

table=[]
for sample in root.findall('url:Sample', ns):
    name = sample.get('iid')
    sup = sample.find('url:Supplementary-Data', ns)
    sup_type = sup.get('type')
    if sup_type == 'BED': 
        for channel in sample.find('url:Channel',ns):
            if channel.get('tag') == 'chip antibody':
                tag = channel.text
        row = (name, sup.text.strip(), tag.strip())
        table.append(row)
df = pd.DataFrame(table, columns=['GSM', 'url','chip_antibody'])

df.head()

#because sometimes series dependent for downloads and sometimes sample dependent, this way tries both:

PATH = '../../output/chip/xmls/'
gse = 'GSE49102'
xml = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+gse+'&targ=all&view=full&form=xml'
os.makedirs(PATH, exist_ok=True)

urlretrieve(xml, PATH+gse+'.xml')
tree = ET.parse(PATH+gse+'.xml')
root = tree.getroot()
ns = {'url': 'http://www.ncbi.nlm.nih.gov/geo/info/MINiML'}

series_table = []
for series in root.findall('url:Series',ns):
    for x in series.findall('url:Supplementary-Data', ns):
        sup_type = x.get('type')
        if sup_type == 'BED':
            name = series.get('iid')
            dwld = x.text.strip()
            row = (name, dwld)
            series_table.append(row)
        else:
            for sample in root.findall('url:Sample', ns):
                sup = sample.find('url:Supplementary-Data', ns)
                sup_type = sup.get('type')
                if sup_type == 'BED': 
                    name = series.get('iid')
                    row = (name, sup.text.strip())
                    series_table.append(row)            
sample_table = []
for sample in root.findall('url:Sample', ns):
    name = sample.get('iid')
    title = sample.find('url:Title', ns)
    for channel in sample.find('url:Channel',ns):
        if channel.get('tag') == 'chip antibody':
            tag = channel.text
    row = (name, title.text, tag.strip())
    sample_table.append(row)
    
sampledf = pd.DataFrame(sample_table, columns=['GSM','title','antibody'])
seriesdf = pd.DataFrame(series_table, columns=['GSE','url'])

sampledf.head()

seriesdf.head()

#try again for GSE49511
from urllib.request import urlretrieve
import os
import xml.etree.ElementTree as ET
import pandas as pd

PATH = '../../output/chip/xmls/'
gse = "GSE18643"
xml = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+gse+'&targ=all&view=full&form=xml'
os.makedirs(PATH, exist_ok=True)

urlretrieve(xml, PATH+gse+'.xml')
tree = ET.parse(PATH+gse+'.xml')
root = tree.getroot()
ns = {'url': 'http://www.ncbi.nlm.nih.gov/geo/info/MINiML'}

series_table = []
for series in root.findall('url:Series',ns):
    for x in series.findall('url:Supplementary-Data', ns):
        sup_type = x.get('type')
        if sup_type == 'BED':
            name = series.get('iid')
            dwld = x.text.strip()
            row = (name, dwld)
            series_table.append(row)
        else:
            for sample in root.findall('url:Sample', ns):
                sup = sample.find('url:Supplementary-Data', ns)
                sup_type = sup.get('type')
                if sup_type == 'BED': 
                    name = series.get('iid')
                    row = (name, sup.text.strip())
                    series_table.append(row)            
sample_table = []
for sample in root.findall('url:Sample', ns):
    name = sample.get('iid')
    title = sample.find('url:Title', ns)
    for channel in sample.find('url:Channel',ns):
        if channel.get('tag') == 'chip antibody':
            tag = channel.text
    row = (name, title.text, tag.strip())
    sample_table.append(row)
    
sampledf = pd.DataFrame(sample_table, columns=['GSM','title','antibody'])
seriesdf = pd.DataFrame(series_table, columns=['GSE','url'])

sampledf.head()

seriesdf.head()

series_table

#try again for GSM463297
#add in TAR
#now i'm just changing everything again... 
#my problem is that every xml is arranged differently, so its hard to know what to take out from where
from urllib.request import urlretrieve
import os
import xml.etree.ElementTree as ET
import pandas as pd

PATH = '../../output/chip/xmls/'
gse = 'GSM463297'
xml = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+gse+'&targ=all&view=full&form=xml'
os.makedirs(PATH, exist_ok=True)

urlretrieve(xml, PATH+gse+'.xml')
tree = ET.parse(PATH+gse+'.xml')
root = tree.getroot()
ns = {'url': 'http://www.ncbi.nlm.nih.gov/geo/info/MINiML'}

sample_table = []
for sample in tree.findall("url:Sample", ns):
    name = sample.get('iid')
    title = sample.find('url:Title', ns) 
    if title is not None:
        print(name, title.text.strip())
    for char in sample.findall('*//url:Characteristics',ns):
        if char.get('tag') == 'chip antibody':
            print(char.text)
for x in tree.findall('*//url:Supplementary-Data', ns):
    sup_type = x.get('type')
    if sup_type == 'BED':
        dwld = x.text.strip()
        row = (dwld, sup_type)
        print(row)
        #_table.append(row)
    else:
        if sup_type == 'TAR':
            dwld = x.text.strip()
            row = (dwld, sup_type)
            print(row)
            #_table.append(row)  
            
for series in tree.findall("url:Series", ns):
    name = series.get('iid')
    title = series.find('url:Title', ns)
    if title is not None:
        print(name, title.text.strip())
    #print(name, series.find('url:Title', ns))
        

#it would be ideal to be able to pull everything I want out without having to do it in the context of either "sample" 
# or "series"... because that changes every time. 

#name -- this could be samples or series or both? maybe just take all
for sample in tree.findall("url:Sample", ns):
    name = sample.get('iid')
    print(name)
for series in tree.findall("url:Series", ns):
    name = series.get('iid')   
    print(name)
#title -- not all titles related to samples. (this has platform titles)
for x in tree.findall('*//url:Title', ns):
    print(x.text)
#antibody -- this would most likely be sample specific
for x in tree.findall('*//url:Characteristics', ns):
    print(x.get('type'))
#bed/tar -- this is general 
for x in tree.findall('*//url:Supplementary-Data', ns):
    print(x.get('type'))

sample_table = []
for sample in tree.findall("url:Sample", ns):
    print(sample)
    name = sample.get('iid')
    title = sample.find('url:Title', ns)
    sample.findall('*//url:Channel',ns)
    for channel in sample.find('url:Channel',ns):
        print(channel)
        break
        if channel.get('tag') == 'chip antibody':
            tag = channel.text
    #row = (name, title.text, tag.strip())
    sample_table.append(row)
    
sampledf = pd.DataFrame(sample_table, columns=['GSM','title','antibody'])
seriesdf = pd.DataFrame(series_table, columns=['GSE','url', 'data_type'])

