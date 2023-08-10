import pandas as pd
import string as st
import matplotlib
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# load dataset as pandas df
data_file = '../data/180213_cogsci_journal_unprocessed.csv'

df_pd = pd.read_csv(data_file)

# fix the article index column name
df_pd.rename(columns={"Unnamed: 0": "Article Index"}, inplace=True)

# NOTE:
# don't fill the column with garbage data, just remove it
#df_pd = df_pd.fillna("Other") 

# drop all rows where 'article_name' is nan
df_pd = df_pd.dropna(how='all', subset=['article_name'])
# need to realign indices to start at 0
df_pd.index=range(len(df_pd.index))

affiliations = df_pd["author_affiliations"].str.lower()
#columns=['Neuro_Science','Psychology','Philosophy','Anthropology','Linguistics','Artificial Intelligence', 'CS', 'CogSci', 'Other', 'Missing', 'Total']
# hexaCount=pd.DataFrame(0, index=np.arange(len(affiliations)), columns=columns)

aff_df = df_pd["author_affiliations"].str.lower()
affiliations = affiliations.tolist()
#aff_df
#df_pd["author_affiliations"]
#print(type(hexaCount[1]))
#hexaCount

# CSV file where each row is a publication type.
# first element of every row is the name of the type

art_types = '../data/Article_Types_V1.csv'
df_art_types = pd.read_csv(art_types, sep=',', header= None)


lst=[[] for _ in range(len(df_art_types))] # makes a list of lists where each sublist is an article type. elem 0 is name, rest elements

#populates list with csv data
for i in range(df_art_types.shape[1]):
        for j in range(df_art_types.shape[0]):
            lst[j].append(df_art_types[i][j])
#removes nan from list of list (it is square by default)           
for a in range(len(lst)):
    lst[a] = [x for x in lst[a] if str(x) != 'nan']

#This function will replace all the inconsistent types of our data set with consolidated types of interest
def ConsolidateType(lst,df_pd):

    oldTypes = df_pd.article_type.values
    newTypes = []
    for i in range(len(oldTypes)):
        # NOTE: index 0 of each sublist of lst is the name of that publication type, not a filter term
        
        for j in range(len(lst)):
            
            if oldTypes[i] in lst[j][1:]:
                
                newTypes.append(lst[j][0])
                break
            elif j == len(lst)-1:
                
                newTypes.append(np.nan)
                
                
    df_pd.article_type = newTypes

ConsolidateType(lst,df_pd)

# removes unwanted types from our dataset
df_pd = df_pd.dropna(how='any', subset=['article_type'])

df_pd.to_csv("../data/Consolidated_Types.csv")

# This cell will make arrays of hexagonal terms from a csv file
aff_types = pd.read_csv("../data/Affiliation_types_V2.csv", sep=',', header=None,encoding='latin-1')

aff_lst=[[] for _ in range(len(aff_types))] # makes a list of lists where each sublist is a hexagonal type. elem 0 is name, rest elements

#populates list with csv data
for i in range(aff_types.shape[1]):
        for j in range(aff_types.shape[0]):
            aff_lst[j].append(aff_types[i][j])
#removes nan from list of list (it is square by default)           
for a in range(len(aff_lst)):
    aff_lst[a] = [x for x in aff_lst[a] if str(x) != 'nan']
    
#aff_df[i] in aff_lst[j][1:]
#if any(x in str for x in a):
#any(x in aff_lst[0][0:] for x in aff_d

columns = []
for cat in aff_lst:
    columns.append(cat[0])

columns.append('other')
columns.append('missing')
columns.append('total')

#This function will replace all the inconsistent types of our data set with consolidated types of interest
def Consolidate_hex(aff_lst,aff_df, columns):

     # initialize empty hexacount df
    hexaCount=pd.DataFrame(0, index=np.arange(len(aff_df)), columns=columns)

    for i in range(len(aff_df)):
        # NOTE: index 0 of each sublist of lst is the name of that publication type, not a filter term
        if pd.isnull(aff_df.iloc[i]):
            # skip because it's nan
            hexaCount["missing"].iloc[i] = 1
            continue
        
        for j in range(len(aff_lst)):
            for k in range(len(aff_lst[j])):
                
                if aff_lst[j][k] in aff_df.iloc[i]:                    
                    
                    hexaCount.iloc[i][aff_lst[j][0]] += 1
                    continue
        
        # append 1 to other if the row is all 0s here
        if hexaCount.iloc[i].sum()<1.:            
                hexaCount["other"].iloc[i] = 1
            
#                 elif j == len(lst)-1:
#                     print(aff_lst[j])
#                     hexaCount['Other'].iloc[j] += 1
        hexaCount["total"].iloc[i] = hexaCount.iloc[i].sum()

    return hexaCount        

# Build Hexagon
hexaCount = Consolidate_hex(aff_lst, aff_df, columns)

hexaCount.iloc[759]

pd.options.display.max_colwidth = 200
#print(columns)
query_hex = columns[3]
cols_to_disp=['author_affiliations']
#print('Querying:', query_hex, '\n---\n')
#print(df_pd[cols_to_disp].loc[hexaCount[query_hex]==1].to_string())

pd_concat = pd.concat([df_pd, hexaCount], axis=1)

pd_concat

#pd_concat[['year'] + columns]

# plot hexagon normalized by total (0:-2 indexing to exclude columns missing and total)
hex_normed = pd_concat[pd_concat['year']>2000][columns[:-2]].div(pd_concat['total'], axis='rows')
hex_normed.sum().plot(kind='bar', figsize=(6,6))
plt.xticks(rotation=90)
plt.title('Sum over all years')

# columns = ['anthropology', 'artificial intelligence', 'linguistics',
#        'neuroscience', 'philosophy', 'psychology', 'computer science', 'cognitive science']

columns = ['linguistics','philosophy', 'psychology', 'computer science', 'neuroscience','anthropology']


print(hex_normed.keys())
# plt.polar(theta_coor,hex_collapsed)
# plt.set_rticks(theta_coor)

hex_collapsed = hex_normed.sum()[columns]/hex_normed.sum()[columns].sum()

# create the radial points
theta = np.linspace(0, np.pi*2-np.pi*2/len(hex_collapsed), len(hex_collapsed))+np.pi/6
theta_app = np.append(theta, theta[0])
theta_app

# plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, polar=True)
ax.plot(theta_app, np.append(hex_collapsed, hex_collapsed[0]), '-o', lw=2)
# plot uniform distribution
ax.plot(theta_app, np.ones(len(theta)+1)/len(theta), 'k--',lw=2)
ax.set_xticks(theta)
ax.set_xticklabels(hex_collapsed.keys(), fontsize=15)

hex_normed.groupby(pd_concat['year']).size()

# this plots sum per year, which means it sums to the total # of articles written that year
#hex_normed.groupby(pd_concat['year']).sum().plot(figsize=(12,6))
hex_normed.groupby(pd_concat[pd_concat['year']>2000]['year']).sum().plot(figsize=(12,6))
plt.title('Hexagon normed, and total by year')

# this plots mean per year, which means it normalizes by the number of articles written per year
#hex_normed.groupby(pd_concat['year']).mean().plot(figsize=(12,6))
#hex_normed.groupby(pd_concat['year']).mean().plot(kind='bar', stacked=True, figsize=(12,6))
hex_normed[columns].groupby(pd_concat[pd_concat['year']>2000]['year']).mean().plot(kind='bar', stacked=True, figsize=(12,6))

#plt.yscale('log')
plt.title('Hexagon normed, and proportion per year')

# pd.groupby is a function that collects rows based on some criteria, in this case, the year of the article
# you guys can figure out how to groupby decades or per 5 years

# Anthropology
#pd_concat[pd_concat.anthropology > 0].article_name.values

# Artificial Intelligence
#pd_concat[pd_concat["artificial intelligence"] > 0].article_name.values

# psychology
#pd_concat[pd_concat.psychology > 0].article_name.values

# philosophy
#pd_concat[pd_concat.philosophy > 0].article_name.values

# Linguistics
#pd_concat[pd_concat.linguistics > 0].article_name.values

# neuroscience
#select rows where anthro # > 0 -> article name
#pd_concat[pd_concat.neuroscience > 0].article_name.values

# OTHER
#aff_df[pd_concat.other > 0]
pd.options.display.max_rows = 1640
len(pd_concat[pd_concat.other > 0][pd_concat.year>2000].author_affiliations.to_frame())

#pd_concat.columns

pd_concat.iloc[1271]

