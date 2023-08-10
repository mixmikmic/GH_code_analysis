get_ipython().run_line_magic('matplotlib', 'inline')
# Overhead
import MySQLdb
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from modules import *

conn = dbConnect()
people = dbTableToDataFrame(conn,'cb_people')
degrees = dbTableToDataFrame(conn,'cb_degrees')
funding_rounds = dbTableToDataFrame(conn,'cb_funding_rounds')
objects = dbTableToDataFrame(conn,'cb_objects')
conn.close()

companies = objects[objects.entity_type=='Company'][['name','entity_id','funding_total_usd','closed_at']].copy()
# Only select companies who have raised some money
companies = companies.dropna(subset=['funding_total_usd'])
display(companies.sort_values('funding_total_usd',ascending=False).head(10))

comp_tot = len(companies)
print('There are {:,} companies in the dataset'.format(comp_tot))

fig,ax = plt.subplots(figsize=(20,20))
n_top_companies = 100
plotset = companies.sort_values('funding_total_usd',ascending=False).head(n_top_companies)
ax = sns.barplot(x='funding_total_usd',y='name',data=plotset)
ax.set_title('Funding Totals for the Top {} Companies'.format(n_top_companies),fontsize=22)
ax.set_xscale('log')
ax.set_xlabel('Funding Total [USD] -- Log Scale',fontsize=18);
ax.set_yticklabels(labels=plotset.name)
ax.set_ylabel('Company',fontsize=18)
plt.savefig('results/funding_totals.png')

I,F = 3,10
for i in np.logspace(I,F,F-I+1):
    num_above = len(companies[companies['funding_total_usd']>i])
    frac = num_above/comp_tot
    print('{:,} companies ({}%) above ${:,}'.format(num_above,round(100*frac,1),int(i)))

people_key = people[['affiliation_name','first_name','last_name','object_id']]
display(people_key.head(10))

ppl_tot = len(people_key)
print('There are {:,} people in the dataset'.format(ppl_tot))

company_people = pd.merge(companies,people_key,left_on='name',right_on='affiliation_name').drop('affiliation_name',axis=1)
display(company_people.head())

cp_tot = len(company_people)
print('There are {:,} people affiliated with funded companies in the dataset'.format(cp_tot))

edu = degrees[['object_id','institution','degree_type','subject']].dropna(subset=['institution'])
CPE = pd.merge(company_people,edu,on='object_id')
display(CPE.head(10))

successful_CPE = CPE[CPE['funding_total_usd']>10_000]
display(successful_CPE.head())

n_entries = len(CPE)
n_unique_ppl = CPE['object_id'].nunique()
print('There are {:,} entries in the dataset, and {:,} unique people in the dataset.'.format(n_entries,n_unique_ppl))
print('On average, successful businesses involve people with an average of {} degrees.'.format(round(n_entries/n_unique_ppl,1)))

def identify_institutions(institutions):
    '''
    Identify institutions using a common name.
    
    Each person in the database has a university which was input by a user. 
    These inputs are not uniform, and so names are inconsistent for some schools.
    (e.g. Berkeley could be UC Berkeley, Cal, or The University of California)
    
    Parameters
    ----------
    institutions : Series
        A series giving a list of institutions to be identified.
        
    Returns
    -------
    inst : Series
        A series where insitutions have been commonly identified.
    
    '''
    ST = SchoolTools()
    # Make all words lowercase
    inst=institutions.str.lower()
    #Remove stopwords
    inst = ST.remove_stopwords(inst)
    # Remove punctuation
    inst = ST.remove_punctuation(inst)
    # Replace common stems with just the stem, delete others, and replace keywords and nicknames with a common title.
    inst = ST.identify_schools(inst)
    return inst.str.title() 

cpe = CPE.copy()
cpe.institution=identify_institutions(cpe.institution)

people_contributed_min = 15
inst_counts = pd.DataFrame(cpe.institution.value_counts())
inst_counts = inst_counts.rename(columns={'institution':'total'})
inst_mult = inst_counts[inst_counts.total>=people_contributed_min]
cpe_count = pd.merge(cpe,inst_mult,left_on='institution',right_index=True).sort_values('total',ascending=False)
cpe_count.head()

fig,edu_ax = plt.subplots(figsize=(16,25))
fig.suptitle('Graduates contributed to the most successful businesses by college',fontsize=22,x=0.55,y=1.01)

edu_ax = sns.barplot(x='total',y='institution',data=cpe_count)
edu_ax.tick_params(labelsize=18)
edu_ax.xaxis.tick_top()
edu_ax.set_xlabel('Number of graduates',fontsize=18);
edu_ax.xaxis.set_label_position('top') 
edu_ax.set_yticklabels(labels=cpe_count.institution.unique(),fontsize=12)
edu_ax.set_ylabel('School',fontsize=18)

fig.tight_layout()
plt.savefig('results/affiliates_by_school.png')

cum_frac = 0
for top_school in ['Stanford','Harvard','Berkeley']:
    frac = inst_counts.total.loc[top_school]/inst_counts.total.sum()
    cum_frac += frac
    print('{} has {}% of the total graduates in top businesses.'.format(top_school,round(100*frac,1)))
print('Together, these three schools produced {}% of the total graduates in top businesses.'.format(round(100*cum_frac,1)))

