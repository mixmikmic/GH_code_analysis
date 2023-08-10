import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.2f' % x)

dtype = {
    'Title': str,
    'First Name': str,
    'Last Name': str,
    'Speciality': str,
    'Institution Name': str
}
df = pd.read_csv('./data/payments.csv', dtype=dtype)

print "{:,} rows".format(len(df))
print "£{:,} total declared".format(df['Amount'].sum())
print len(df['Company Name'].unique()), 'companies'
print len(df['Organisation Name'].unique()), 'organisations'
print len(df['Speciality'].unique()), 'specialities'
print len(df['Institution Name'].unique()), 'institutions'
# df.describe(include='all')

df_hcps = df[pd.isnull(df['Organisation Name'])]
df_hcos = df[~pd.isnull(df['Organisation Name'])]
a = "{:,} payments to HCOs ({:,.2f}% of total payments)"
print a.format(len(df_hcos), 100 * len(df_hcos) / float(len(df)))
a = "{:,} payments to HCPs ({:,.2f}% of total payments)"
print a.format(len(df_hcps), 100 * len(df_hcps) / float(len(df)))
print 

a = "£{:,} paid to HCOs ({:,.2f}% of total payments)"
print a.format(df_hcos['Amount'].sum(), 100 * df_hcos['Amount'].sum() / df['Amount'].sum())
a = "£{:,} paid to HCPs ({:,.2f}% of total payments)"
print a.format(df_hcps['Amount'].sum(), 100 * df_hcps['Amount'].sum() / df['Amount'].sum())

# This is what the raw payments data looks like!
df.head()

print df['Amount'].describe()
print len(df[df['Amount'] > 10000]), 'payments overall are more than £10,000'

print '\nPayments to organisations:'
print df_hcos['Amount'].describe()
print '\nPayments to individuals:'
print df_hcps['Amount'].describe()

df_hcps.sort_values('Amount', ascending=False).head()

df.sort_values('Amount').head()[['Company Name', 'Organisation Name', 'TOV Type', 'Amount']]

df.groupby(('TOV Category', 'TOV Type')).sum().sort_values('Amount', ascending=False)

df_aggregate = pd.read_csv('./data/aggregates.csv')
df_aggregate.head()

undeclared_payments = df_aggregate['Amount'].sum()
declared_payments = df['Amount'].sum()
total_payments = undeclared_payments + declared_payments

print 'Total payments undeclared:', "£{:,}".format(undeclared_payments)
print 'Total payments declared:', "£{:,}".format(declared_payments)
print 'Percentage undeclared: {:.2f}%'.format(undeclared_payments / total_payments * 100)

df_agg_ex_rd = df_aggregate[df_aggregate['TOV Category'] != 'Research & Development']
undeclared_payments_ex_rd = df_agg_ex_rd['Amount'].sum()
print 'Total payments on R&D (none of which are declared):', "£{:,}".format(undeclared_payments - undeclared_payments_ex_rd)
print 'Total payments undeclared ex R&D:', "£{:,}".format(undeclared_payments_ex_rd)

# Get aggregate figures for HCPs only. 
df_agg_hcps = df_agg_ex_rd[pd.isnull(df['Organisation Name'])].groupby(('Company Name')).sum().reset_index()
df_agg_hcps = df_agg_hcps[['Company Name', 'Amount', 'No of HCP HCO in Aggregate']]
df_agg_hcps.rename(columns={'No of HCP HCO in Aggregate': 'No of HCP'}, inplace=True)
# print df_agg_hcps.head()

# Count total undeclared payments by company/category/HCP
declared_hcps = df_agg_hcps['No of HCP'].sum()
print 'No of HCP payments by company/category, undeclared:', declared_hcps #
undeclared_hcps = len(df_hcps)
print 'No of HCP payments by company/category, declared:', undeclared_hcps
v = undeclared_hcps / (undeclared_hcps + declared_hcps) 
print 'Overall % of payments to HCPs undeclared: {:.2f}%'.format(v*100)

# Group the declared HCP payments by company, then merge with the aggregate data.
df_hcps_by_co = df_hcps.groupby('Company Name')['Amount'].agg(['sum','count']).reset_index()
df_hcps_by_co.rename(columns={'sum': 'Amount', 'count': 'No of HCP'}, inplace=True)
df_m = pd.merge(df_agg_hcps, df_hcps_by_co, on='Company Name', suffixes=(' Undeclared', ' Declared'))
df_m['Total HCPs'] = df_m['No of HCP Undeclared'] + df_m['No of HCP Declared']
df_m['% HCP Payments Undeclared'] = df_m['No of HCP Undeclared'] / df_m['Total HCPs'] * 100
df_m.sort_values('% HCP Payments Undeclared', ascending=False)

# Get all aggregate payments but EXCLUDE R&D. 
# This is because NO R&D payments are included in the payments sheet, so 
# including them in our per-company analyses will distort results.  
# Lump both HCPs and HCOs together.
df_agg_by_co = df_agg_ex_rd.groupby(('Company Name')).sum().reset_index()
df_agg_by_co.rename(columns={'No of HCP HCO in Aggregate': 'No of HCP HCO'}, inplace=True)
df_agg = df_agg_by_co[['Company Name', 'Amount', 'No of HCP HCO']]

# We've already created dataframes for HCOs and HCPs: now get the sums
# and counts of these, and merge them. There's probably a better way to do this. 
df_hcos_by_co = df_hcos.groupby('Company Name')['Amount'].agg(['sum','count']).reset_index()
df_hcps_by_co = df_hcps.groupby('Company Name')['Amount'].agg(['sum','count']).reset_index()
df_m = pd.merge(df_hcos_by_co, df_hcps_by_co, on='Company Name', suffixes=('_hco', '_hcp'))
df_m['Amount'] = df_m['sum_hco'] + df_m['sum_hcp']
df_m['No of HCP HCO'] = df_m['count_hco'] + df_m['count_hcp']
df_declared = df_m[['Company Name', 'Amount', 'No of HCP HCO']]

df_combined = pd.merge(df_declared, df_agg, on='Company Name', 
                        suffixes=(' Declared', ' Undeclared'))

# Calculate totals, proportions etc. 
df_combined['Total Amount'] = df_combined['Amount Declared'] + df_combined['Amount Undeclared']
df_combined['Total HCOs/HCPs'] = df_combined['No of HCP HCO Declared'] + df_combined['No of HCP HCO Undeclared']
df_combined['Proportion of names undeclared'] = df_combined['No of HCP HCO Undeclared'] /     df_combined['Total HCOs/HCPs'] * 100
df_combined['Proportion of total amount undeclared'] = df_combined['Amount Undeclared'] /     df_combined['Total Amount'] * 100
    
# Print summary stats about the proportion of undeclared payments.
# Note that this number is lower than the 77% above, because we've excluded R&D. 
print 'Total % names undeclared:', "{:.2f}%".format(df_combined['No of HCP HCO Undeclared'].sum() /     df_combined['Total HCOs/HCPs'].sum() * 100)
print 'Total % amount undeclared:', "{:.2f}%".format(df_combined['Amount Undeclared'].sum() /     df_combined['Total Amount'].sum() * 100)

# Show all companies for which total amount > £1 million,
# sorted by the total amount undeclared.
# Again, these are flattering numbers because we've excluded R&D.
# In most cases, it looks as though the payments that are undeclared are the 
# smaller payments - i.e. probably those to individuals. Sanofi seems to be an exception.
# Note how GSK is an outlier too. 
df_combined.sort_values('Proportion of total amount undeclared', inplace=True, ascending=False)
df_combined[df_combined['Total Amount'] > 1000000]

# Double-check a sample row in the table above, against the raw data,
# to make sure we've got things right....
print df_agg_by_co[df_agg_by_co['Company Name'] == 'Napp Pharmaceuticals Ltd'][['Amount', 'No of HCP HCO']]
print df[df['Company Name'] == 'Napp Pharmaceuticals Ltd'].sum()['Amount']
print len(df_hcps[df_hcps['Company Name'] == 'Napp Pharmaceuticals Ltd']['Last Name'])
print len(df_hcos[df_hcos['Company Name'] == 'Napp Pharmaceuticals Ltd']['Organisation Name'])

# Save all data to CSV.
df_combined.to_csv('declared_vs_aggregated_payments_by_company.csv')

# As above, EXCLUDE R&D. 
df_agg_by_cat = df_agg_ex_rd.groupby('TOV Category').sum().reset_index()
df_agg_cat = df_agg_by_cat[['TOV Category',  'Amount']]
df_by_cat = df.groupby(('TOV Category')).sum().reset_index()
df_merged_cat = pd.merge(df_agg_cat, df_by_cat, on=['TOV Category'], 
                     suffixes=(' Aggregate', ' Declared'))
df_merged_cat['Total Amount'] = df_merged_cat['Amount Declared'] +     df_merged_cat['Amount Aggregate']
df_merged_cat['Proportion Undeclared'] = df_merged_cat['Amount Aggregate'] /     df_merged_cat['Total Amount']
df_merged_cat.sort_values('Proportion Undeclared', ascending=False, inplace=True)
df_merged_cat.to_csv('declared_vs_aggregated_payments_by_category.csv')
df_merged_cat

df_individual = df.groupby(('Last Name', 'First Name', 'Speciality')).sum().reset_index()
df_individual.sort_values(by='Amount', ascending=False).head(20)

df_by_co = df.groupby('Company Name').agg(['sum', 'mean', 'median', 'count'])
df_by_co.sort_values(by=('Amount', 'sum'), ascending=False).head(10)

#  df[['Company Name', 'TOV Category', 'Amount']].head()
df.groupby(('Company Name', 'TOV Category')).sum().sort_values('Amount', ascending=False).head(10)

df_by_co_and_speciality = df.groupby(('Company Name', 'Speciality')).sum()    .sort_values('Amount', ascending=False).reset_index()

total = df_by_co_and_speciality.groupby('Company Name')['Amount'].transform('sum')
df_by_co_and_speciality['% of Co Spend'] = df_by_co_and_speciality['Amount']/total * 100

print df_by_co_and_speciality.head(10)
# Validate % calculation
# print df_by_co_and_speciality[df_by_co_and_speciality['Company Name'] == 'AstraZeneca']['% of Company Total Spend'].sum()
df_by_co_and_speciality.to_csv('payments_by_co_and_speciality.csv')

df_by_org = df.groupby('Organisation Name').agg(['sum', 'mean', 'median', 'count'])
df_by_org.sort_values(by=('Amount', 'sum'), ascending=False).head(10)

df_by_spec = df.groupby('Speciality').agg(['sum', 'mean', 'median', 'count']).sort_values(by=('Amount', 'sum'), ascending=False)
print df_by_spec.head(10)
df_by_spec.to_csv('by_speciality.csv')

# Normalise the specialties
def specialty_classifier(row):
    other = ['Healthcare Administration',
             'Microbiology',
             'Laboratory - Medical Analysis',
             'Research',
             'Occupational Therapist',
             'Wholesaler', 
             'Miscellaneous']
    pharmacists = ['Clinical Pharmacology', 'Pharmacist']
    nurses = ['Nurse']
    classification = 'HCO'
    if row['Speciality'] in other:
        classification = 'Other'
    elif row['Speciality'] in pharmacists:
        classification = 'Pharmacists'
    elif row['Speciality'] in nurses:
        classification = 'Nurses'
    elif str(row['Speciality']) != "nan":
        # XXX there must be a nicer way to filter out pd.nan values...
        classification = 'Doctors'
    return classification

df['Normalised specialty'] = df.apply(specialty_classifier, axis=1)

# Re-run the analysis   
df.groupby('Normalised specialty').agg(['sum', 'mean', 'median', 'count']).sort_values(by=('Amount', 'sum'), ascending=False)

df_individual = df.groupby(('Last Name', 'First Name', 'Speciality')).sum().reset_index()
df_individual.groupby('Speciality').count()[['Amount']].sort_values("Amount", ascending=False)

df_individual = df.groupby(('Last Name', 'First Name', 'Normalised specialty')).sum().reset_index()
df_individual.groupby('Normalised specialty').count()[['Amount']].sort_values("Amount", ascending=False)

df_sample = df[(df['Company Name'] == 'Genzyme') & (df['TOV Type'] == 'Fees')]
df_sample.describe(include='all')
# print len(df_sample), 'rows'
# df_sample['Amount'].sum()






get_ipython().magic('pylab inline')
plt.figure()
bin_range = np.arange(0, 1100000, 10000)
df['Amount'].plot(kind='hist', bins=bin_range, color='blue', alpha=0.6)
plt.ylabel("Number of payments")
plt.xlabel("Payment size")
plt.yscale('log', nonposy='clip')
plt.title("Payments by amount")
plt.grid()



