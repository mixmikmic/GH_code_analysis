import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cdf = pd.read_csv('../data/311_sample_head.csv')

mn = pd.read_csv('../data/BORO_zip_files_csv/MN.csv')
qn = pd.read_csv('../data/BORO_zip_files_csv/QN.csv')
bk = pd.read_csv('../data/BORO_zip_files_csv/BK.csv')
si = pd.read_csv('../data/BORO_zip_files_csv/SI.csv')
bx = pd.read_csv('../data/BORO_zip_files_csv/BX.csv')
bdf = pd.concat([mn,qn,bk,si,bx])

bdf.head()

y[:-1]/cdf.shape[0]

y = cdf.Borough.value_counts()[:-1] / cdf.shape[0]
x = y.index
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align='center', alpha=0.5)
plt.yticks(y_pos, x)
plt.xlabel('Fraction of Complaints')
plt.title('Complaints by Borough, Before Processing')

def clean_address(address):
    '''
    Strip whitespace and convert string to upper
    '''
    try:
        return str.upper(address).strip()
    except:
        return None

def clean(df):
    '''
    Clean and filter the building dataframe
    Use only relevant columns and remove rows where YearBuilt==0
    '''
    df = df[['Address','XCoord','YCoord','ZipCode','YearBuilt','Borough']].copy()
    df['Address'] = df['Address'].apply(clean_address)
    df.drop(df[df['YearBuilt']==0].index,inplace=True)
    return df

def summary(df):
    '''
    Print percent null and top 10 most common values
    '''
    pct_null = df['Address'].isnull().sum()*1./df.shape[0]
    print "pct null:",round(pct_null,2)
    print df['Address'].value_counts().head(10)

def import_neighborhoods(fname):
    '''
    import wikipedia neighborhoods file
    returns: dictionary where key is borough
        and value is list of neighborhoods
    '''
    with open(fname,'r') as f:
        raw = f.read()
    lines = raw.split('\n')
    boroughs = {}
    for line in lines:
        fields = line.split(',')
        borough = fields.pop(0)
        if borough not in boroughs:
            boroughs[borough]=[]
        for f in fields:
            if f:
                neighborhood = str.upper(f.strip())
                boroughs[borough].append(neighborhood)
    boroughs2 = {'QN':boroughs['Queens']+['QUEENS'],
                 'BK':boroughs['Brooklyn']+['BROOKLYN'],
                 'MN':boroughs['Manhattan']+['MANHATTAN','NEW YORK'],
                 'SI':boroughs['Staten Island']+['STATEN ISLAND'],
                 'BX':boroughs['Bronx']+['BRONX']}
    return boroughs2

def city2borough(city):
    for borough,hood_list in neighborhoods.iteritems():
        try:
            if str.upper(city) in hood_list:
                return borough
        except:
            return None

#clean data
bdf = clean(bdf)

#clean up and rename address
cdf['Address'] = cdf['Incident Address'].apply(clean_address)

#import neighborhoods data
hoods_fname='../data/wiki_Neighborhoods_in_New_York_City.csv'
neighborhoods = import_neighborhoods(hoods_fname)
cdf['orig_borough'] = cdf['Borough'].copy()
cdf['Borough'] = cdf.City.apply(city2borough)
hdf = cdf[cdf['Complaint Type']=='HEATING']

pd.crosstab(cdf.Borough,cdf.orig_borough)

cdf.Borough.value_counts()

y = cdf.Borough.value_counts() / cdf.shape[0]
x = y.index
x = ['BROOKLYN','BRONX','QUEENS','MANHATTAN','STATEN ISLAND']
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align='center', alpha=0.5)
plt.yticks(y_pos, x)
plt.xlabel('Fraction of Complaints')
plt.title('Complaints by Borough, After Neighborhood Mapping')

#Remove duplicates from buildings dataframe.  Only keep the newest one
bdf = bdf.sort_values(by='YearBuilt',ascending=False)
dupes = bdf.duplicated(subset=['Borough','Address'],keep='first')
bdf.drop(dupes[dupes].index,inplace=True)

#Merge buildings dataframe with heating dataframe
mdf = hdf.merge(bdf,how='inner',on=['Address','Borough'])

mdf2 = hdf.merge(bdf,how='left',on=['Address','Borough'])

print mdf.shape
print hdf.shape
print bdf.shape

hdf.head()

mdf.shape[0]*1./hdf.shape[0]-1

counts = mdf.groupby(['YearBuilt']).count()

plt.scatter(counts.index,counts['Complaint Type'])
plt.title("Complaints By Building Completion Date")
plt.xlabel("Completion Date (Year Built) of Buildings")
plt.ylabel("Number of Complaints")

first_year = 1880
bdf.loc[bdf['YearBuilt']>=first_year,'YearBuilt'].hist(bins=2016-first_year)
plt.title('Buildings by Age')

building_counts = bdf.groupby('YearBuilt').count()['Borough']

counts['building_count']=building_counts

plt.scatter(counts.index,counts['Complaint Type']/counts['building_count'])
plt.xlim([1880,2016])
plt.title("Complaints per Building by Year")
plt.xlabel("Year Built")
plt.ylabel("Complaints per Building")



age = pd.read_csv('../data/complaints vs. age.csv')

age.head()

age['complaints_per_building'] = age.complaints/age.total

plt.figure(figsize=(7,5))
plt.scatter(age.YearBuilt,age.complaints_per_building)
plt.xlim([1820,2016])
plt.title("Complaints per Building by Building Age")
plt.xlabel("Year Built")
plt.ylabel("Avg. Num. Complaints per Building")



