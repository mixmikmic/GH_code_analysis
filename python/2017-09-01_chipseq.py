import tempfile
import pandas as pd
import urllib
import os
from urllib.request import urlopen

url = 'http://intermine.modencode.org/release-33/features.do?type=experiment&action=export&experiment=Chromatin%20Binding%20Site%20Mapping%20of%20Transcription%20Factors%20in%20D.%20melanogaster%20by%20ChIP-seq&feature=BindingSite&format=csv'    
response = urlopen(url)
    
with tempfile.NamedTemporaryFile() as temp:     
    temp.write(response.read())
    temp.seek(0)
    df = pd.read_csv(temp)

df.columns = ['DB_identifier','score','chrom','start','end','strand','modENCODE_id','pivot','name']

df.head()

happy = df.set_index(
    ['modENCODE_id','DB_identifier','score','chrom','start','end','strand','pivot']).pivot_table(
        columns='pivot', index=['DB_identifier','score','chrom','start','end','strand','modENCODE_id'], 
        values='name',aggfunc='first')

happy.reset_index().head()

happy[(happy['cell line'] == 'S2-DRSC')].modENCODE_id.unique()

happy[happy['cell line'] == 'None'].modENCODE_id.unique()

happy['cell line'].unique()

happy.describe()

happy=happy.reset_index()

happy.modENCODE_id.unique()

happy.modENCODE_id.describe()

cnt = happy.groupby(['modENCODE_id']).agg({'start':'count'})
cnt.start.describe()

phantompeaks = pd.read_excel('/Users/bergeric/Downloads/gkv637_Supplementary_Data/Supplementary_table_3__List_of_Phantom_Peaks.xlsx')

phantom_overlap = pd.read_excel('/Users/bergeric/Downloads/gkv637_Supplementary_Data/Supplementary_table_5__Overlap_of_the_Phantom_Peaks_with_non-histone_modENCODE_ChIPSeq_profiles.xlsx', header=1)

phantom_overlap.head()

new_ids = []
for val in list(phantom_overlap.modE_ID):
    newval = 'modENCODE_'+str(val)
    new_ids.append(newval)

len(new_ids)

len(list(phantom_overlap.modE_ID))

phantom_overlap['modENCODE_id'] = new_ids

phantom_overlap.head()

overlapping = happy.merge(phantom_overlap, on='modENCODE_id', how='inner')

phantom_overlap.shape

overlapping.modENCODE_id.describe()

phantompeaks.head()

phantom_overlap.modENCODE_id.describe()

phantom_overlap.merge(happy, on='modENCODE_id', how='left').head()

count=0
for val in list(happy.modENCODE_id.unique()): 
    if val in list(phantom_overlap.modENCODE_id):
        count += 1 
print(count)

missing = []
for val in list(phantom_overlap.modENCODE_id):
    if val not in list(happy.modENCODE_id.unique()):
        missing.append(val)

len(missing)

blob = pd.Series(missing)

url = 'http://data.modencode.org/cgi-bin/cloud_list.pl?accessions=3954,3393,3806,3825,3231,2625,2626,2637,3403,4078,3240,4080,5068,4082,4081,3959,5069,2638,2639,3395,3235,4974,5008,5070,5071,5072,5577,3229,3230,3402,3401,2640,2641,2642,3234,3236,3239,3241,3400,3398,4976,3824,3826,4089,3809,3238,3397,5028,3814,3245,3830,4119,4981,5257,5073,5074,5595,5596,5075,4953,5258,5076,5077,5078,5079,5080,5081,5082,4946,5083,4951,4960,4120,5084,5578,844,845,834,837,838,839,835,836,840,841,895,842,843,5085,5086,5087,5088,5582,3807,3810,3811,3815,3955,5089,5583,3829,820,810&urls=1'
page = urlopen(url).read().decode('utf-8')
page = page.split('\n')

table = []
for line in page:
    if not line.startswith('#'):
        modid = 'modENCODE_'+str(line.split()[0]) #gives an error but still works? 
        link = line.split()[1]
        table.append([modid, link])

download_urls = pd.DataFrame(table, columns=['modENCODE_id', 'link'])

download_urls.head()

len(download_urls.modENCODE_id.unique())

download_these = []
for val in list(download_urls.modENCODE_id.unique()):
    if val not in list(happy.modENCODE_id.unique()):
        download_these.append(val)        

len(download_these)

PATH = '../../data/modENCODE_downloads/'
os.makedirs(PATH, exist_ok=True)

for index, row in download_urls.iterrows(): 
    if row['modENCODE_id'] in list(download_these): 
        if 'gff' in row['link']: 
            urllib.request.urlretrieve(row['link'], PATH+row['modENCODE_id']+'.gff.gz')       

#import zipfile
#with zipfile.ZipFile("file.zip","r") as zip_ref:
    #zip_ref.extractall("targetdir")

import glob
import os
for fname in glob.glob('../../data/modENCODE_downloads/modENCODE_*.gff'): 
    name = os.path.splitext(os.path.basename(fname))[0]
    df = pd.read_table(fname, header=None, comment='#')
    df[8] = 'ID='+name
    new = []
    for val in df[0]:
        newval = 'chr'+val
        new.append(newval)
    df[0] = new
    df.to_csv(fname, sep='\t', header=None, index=False)

df.head()

import pybedtools
from pybedtools import BedTool
from pybedtools.featurefuncs import gff2bed

genes = BedTool('../../data/dmel-all-r6.12.gene_only.chr.gff')

genes_bed = genes.each(gff2bed, name_field='ID').saveas()

moden895=pd.read_table('../../data/modENCODE_downloads/modENCODE_895.gff', header=None)
moden895[5] = '.'
moden895[6] = '.'
moden895.to_csv('../../data/modENCODE_downloads/modENCODE_895.gff', sep='\t', header=None, index=False)

#problem file 5070: 
peak = pd.read_table('../../data/modENCODE_downloads/modENCODE_5070.gff', header=None)
peak[4] = peak[4].astype(int)
peak[3] = peak[3].astype(int) 
peak.to_csv('../../data/modENCODE_downloads/modENCODE_5070.gff', sep='\t', header=None, index=False)

#problem file 5084: 
peak = pd.read_table('../../data/modENCODE_downloads/modENCODE_5084.gff', header=None)
peak[4] = peak[4].astype(int)
peak[3] = peak[3].astype(int) 
peak.to_csv('../../data/modENCODE_downloads/modENCODE_5084.gff', sep='\t', header=None, index=False)

#problem file 5583: 
peak = pd.read_table('../../data/modENCODE_downloads/modENCODE_5583.gff', header=None)
peak[4] = peak[4].astype(int)
peak[3] = peak[3].astype(int) 
peak.to_csv('../../data/modENCODE_downloads/modENCODE_5583.gff', sep='\t', header=None, index=False)

#problem file 844: 
peak = pd.read_table('../../data/modENCODE_downloads/modENCODE_844.gff', header=None)
peak[4] = peak[4].astype(int)
peak[3] = peak[3].astype(int) 
peak.to_csv('../../data/modENCODE_downloads/modENCODE_844.gff', sep='\t', header=None, index=False)

#iterate over modENCODE_downloads files to get beds for liftover
PATH = '../../data/modENCODE_downloads/'
concat = []
for fname in glob.glob('../../data/modENCODE_downloads/modENCODE_*.gff'): 
    peaks = BedTool(fname)
    peaks_bed = peaks.remove_invalid().each(gff2bed).saveas(PATH+os.path.splitext(os.path.basename(fname))[0]+'.bed')   

#need to figure out intersect step now that I have liftover files 
#think it might be easier to save giant bed file and do the intersect on that
#instead of in a loop? 

concat = []
for fname in glob.glob('../../data/modENCODE_downloads/modENCODE_*.liftover'):
    if os.path.getsize(fname) > 0 :
        df = pd.read_table(fname, header=None)
        concat.append(df)
    else:
        print(fname)

bigdf = pd.concat(concat)
#save this matrix of bed files
bigdf.to_csv('../../data/modENCODE_downloads/modENCODE_allliftovers', sep='\t', header=None, index=False)
bigdf.head()

bigdf.columns = ['chrom','start','end','id','blank1','blank2']

len(bigdf['id'].unique())

cnts = bigdf.groupby(['id']).agg({'start': 'count'})

cnts.start.describe()

intergenicdf = pd.read_table('../../data/dmel-all-r6.12.chr.intergenic.bed', header=None, names=['chrom','start','end'])

intergenicdf['length'] = (intergenicdf['end'] - intergenicdf['start'])

intergenicdf.length.describe()

#one giant bed intersect using big file "modENCODE_allliftovers" 
with open('../../data/modENCODE_downloads/modENCODE_allliftovers') as f: 
    peaks_bed = BedTool(f)
    intersect = genes_bed.intersect(peaks_bed, u=True).saveas()
    intdf = intersect.to_dataframe()

len(intdf.name.unique())

intdf.head()

happy.to_csv('../../data/modENCODE_downloads/modENCODE_whitelab', sep='\t', header=None, index=False)

happypeaks = BedTool('../../data/modENCODE_downloads/modENCODE_whitelab')
#lose a lot of information during gff2bed step... 
happypeaks_bed = peaks.remove_invalid().each(gff2bed).saveas('../../data/modENCODE_downloads/modENCODE_whitelab.bed')

with open('../../data/modENCODE_downloads/modENCODE_whitelab.liftover') as f: 
    peaks_bed = BedTool(f)
    intersect = genes_bed.intersect(peaks_bed, u=True, wb=True).saveas()
    white = intersect.to_dataframe()

white.head()

allintersects = pd.concat([white, intdf])

allintersects.drop_duplicates().shape

intdf.shape

#Can merge back on start/end to get information lost... should probably do that before
#concat. Although maybe not if super redundant? 

