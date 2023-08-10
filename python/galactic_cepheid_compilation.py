import numpy as np
import pandas as pd
import reddening_laws as red
from scipy.optimize import curve_fit

ir_df = pd.read_csv("cepheids_ir_metallicities.txt", delim_whitespace=True)

ir_df

optical_df = pd.read_csv('mw_ceps_optical_no_ri', delim_whitespace=True)

optical_df

## Removing the underscores in the names in this table so I can match by ID

optical_df['ID'] = optical_df['ID'].str.replace('_', '')

## Add a log P column to be consistent with other dfs
optical_df['logP'] = np.log10(optical_df['Period'])

optical_df['Galaxy'] = 'MW'

optical_df

lmc_df = pd.read_csv('LMC_true_data.dat', delim_whitespace=True)

lmc_df

## Removing redundant column from the data frame
lmc_df = lmc_df.drop('No.', 1)
optical_df = optical_df.drop('Period', 1)

lmc_df

## Renaming the columns to make them consistent
lmc_df.rename(columns={'36':'[3.6]', '45':'[4.5]', 'FeH':'[Fe/H]', 'HV':'ID'}, inplace=True)

lmc_df['Galaxy'] = 'LMC'

lmc_df

frames = [ir_df, optical_df, lmc_df]

## Need to make sure that the ID system is consistent between dataframes

ir_names = pd.unique(ir_df.ID.ravel())
optical_names = pd.unique(optical_df.ID.ravel())
lmc_names = pd.unique(lmc_df.ID.ravel())

# converting from numpy ndarrays to reguar lists so I can do string operations

ir_names = ir_names.tolist()
optical_names = optical_names.tolist()
lmc_names = optical_names.tolist()

all_names = ir_names + optical_names + lmc_names

unique_names = pd.unique(all_names)


print unique_names

## Obviously some repeated here because of capitalisation. Convert all to lowercase so I can do sorting properly

lower_case = map(str.lower, unique_names)
sorted_names = sorted(lower_case)
identical = [x for x in sorted_names if sorted_names.count(x) >= 2]
similar_stars = []
for star in range(len(sorted_names)):
    sim = sorted_names[star][-3:]
    similar = []
    for star2 in range(len(sorted_names)):
        similar.append(sim in sorted_names[star2])
    total = sum(similar)
    if (total >= 2):
        #print total, sorted_names[star]
        similar_stars.append(sorted_names[star])

        
# sort by the last three characters        
sorted(similar_stars, key=lambda x: x[-3:])

ir_df.index.name = 'StarNo'
ir_df

ir_df.ix[15, 'ID'] = 'BetaDor'
ir_df.ix[4, 'ID'] = 'lCar'
ir_df

sum(result['Galaxy']=='MW')

## Now to match - use pd.combine_first
lmc_mw = lmc_df.combine_first(optical_df)

lmc_mw

## Now add in the IR data

full_df = lmc_mw.combine_first(ir_df)

full_df

full_df['Av'] = np.NAN
full_df['Ak'] = np.NAN
full_df['M_V'] = np.NAN
full_df['M_H'] = np.NAN

full_df.set_index('ID', inplace=True, verify_integrity=True)

full_df.index.is_unique

cut_ir_df = ir_df[['ID', 'logP', '[3.6]', '[4.5]', '[Fe/H]', 'Galaxy', 'M_3.6', 'M_4.5']]

cut_optical = optical_df[['ID', 'logP', 'B', 'V', 'J', 'H', 'K', 'Galaxy']]

cut_lmc = lmc_df[['ID', 'logP', 'B', 'V', 'J', 'H', 'K', '[3.6]', '[4.5]', '[Fe/H]', 'Galaxy']]

cut_lmc.set_index('ID', inplace=True, verify_integrity=True)

cut_optical.set_index('ID', inplace=True, verify_integrity=True)
cut_ir_df.set_index('ID', inplace=True, verify_integrity=True)

duplicates_ir = sum(cut_ir_df.duplicated(subset='ID'))
duplicates_lmc = sum(cut_lmc.duplicated(subset='ID'))
duplicates_optical = sum(cut_optical.duplicated(subset='ID'))
## print duplicates_ir, duplicates_lmc, duplicates_optical

## combine_first is doing some weird shit to the data. Use concat or append instead


combined_df = pd.concat([cut_ir_df, cut_optical], axis=1, join_axes=[cut_ir_df.index])

cut_lmc

combined_df.update(cut_lmc)

combined_df

combined_df['Av'] = np.NAN
combined_df['Ak'] = np.NAN
combined_df['M_V'] = np.NAN
combined_df['M_H'] = np.NAN

combined_df.ix['lCar', 'Av'] = 0.52
combined_df.ix['zetaGem', 'Av'] = 0.06
combined_df.ix['BetaDor', 'Av'] = 0.25
combined_df.ix['WSgr', 'Av'] = 0.37
combined_df.ix['XSgr', 'Av'] = 0.58
combined_df.ix['YSgr', 'Av'] = 0.67
combined_df.ix['deltaCep', 'Av'] = 0.23
combined_df.ix['FFAql', 'Av'] = 0.64
combined_df.ix['TVul', 'Av'] = 0.34
combined_df.ix['RTAur', 'Av'] = 0.20

combined_df.ix['lCar', 'Ak'] = 0.06
combined_df.ix['zetaGem', 'Ak'] = 0.01
combined_df.ix['BetaDor', 'Ak'] = 0.03
combined_df.ix['WSgr', 'Ak'] = 0.04
combined_df.ix['XSgr', 'Ak'] = 0.07
combined_df.ix['YSgr', 'Ak'] = 0.07
combined_df.ix['deltaCep', 'Ak'] = 0.03
combined_df.ix['FFAql', 'Ak'] = 0.08
combined_df.ix['TVul', 'Ak'] = 0.02
combined_df.ix['RTAur', 'Ak'] = 0.02

combined_df

## Removing XSgr because it doesn't have a metallicity
combined_df.drop('XSgr', inplace=True)

combined_df

# convert Ak to Ah, calculate M_V and M_H
## H is 1.63 microns

combined_df['Ah'] = np.NaN
combined_df.Ah = red.indebetouw_ir(1.63)*combined_df.Ak

combined_df['mu_LKH'] = np.NAN
combined_df.ix['lCar', 'mu_LKH'] = 8.56
combined_df.ix['zetaGem', 'mu_LKH'] = 7.81
combined_df.ix['BetaDor', 'mu_LKH'] = 7.54
combined_df.ix['WSgr', 'mu_LKH'] = 8.27
combined_df.ix['YSgr', 'mu_LKH']= 8.51
combined_df.ix['deltaCep','mu_LKH'] = 7.19
combined_df.ix['FFAql', 'mu_LKH'] = 7.79
combined_df.ix['TVul', 'mu_LKH'] = 8.73
combined_df.ix['RTAur', 'mu_LKH'] = 8.15

combined_df

## Finally! Get the abs mags

combined_df.M_V = combined_df.V - combined_df.mu_LKH - combined_df.Av
combined_df.M_H = combined_df.H - combined_df.mu_LKH - combined_df.Ah

## Made duplicate columns of galaxy and logP somehow. Removing those by transposing and back
combined_df = combined_df.T.groupby(level=0).first().T

combined_df



combined_df.to_csv('Combined_Cepheid_data.txt', sep='\t', na_rep='99.999', float_format='%.3f', index=True, columns=['logP','B','V','J','H','K','[3.6]','[4.5]','[Fe/H]', 'Galaxy', 'M_3.6', 'M_4.5', 'M_V', 'M_H'])

combined_df

combined_df.to_csv('Combined_Cepheid_data.txt', sep='\t', na_rep='99.999', float_format='%5.3f', index=True, columns=['logP','B','V','J','H','K','[3.6]','[4.5]','[Fe/H]', 'Galaxy', 'M_3.6', 'M_4.5', 'M_V', 'M_H'])

# Not formatting the file correctly because the numbers are stored as objects not floats. Convert them!

combined_df = combined_df.convert_objects(convert_numeric=True)

combined_df.dtypes

combined_df.to_csv('Combined_Cepheid_data.txt', sep='\t', na_rep='99.999', float_format='%5.3f', index=True, columns=['logP','B','V','J','H','K','[3.6]','[4.5]','[Fe/H]', 'Galaxy', 'M_3.6', 'M_4.5', 'M_V', 'M_H'])



