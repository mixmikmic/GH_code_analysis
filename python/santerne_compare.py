import numpy as np
import pandas as pd

fpp_all = pd.read_csv('fpp_final_table.csv', index_col=0)
fpp_all.count()

columns = ['kepid', 'koi', 'kepler_pl', 'period','depth','aR','Kp',
           'Teff','logg','feh','nature','reference']

from keputils.koiutils import koiname
santerne = pd.read_table('TargetSelection.tex', delimiter='\s*&\s*',
                        names=columns)
santerne['koi'] = santerne.koi.apply(koiname)
santerne['FPP'] = fpp_all.ix[santerne.koi,'FPP'].values

santerne.groupby('nature').count()

eb_kois = santerne.query('nature=="EB"').koi
fpp_all.ix[eb_kois,'rprs']

santerne.query('nature=="EB"')



print santerne.groupby('nature')['FPP'].mean()
print santerne.groupby('nature')['FPP'].median()
print santerne.groupby('nature')['FPP'].count()

# nature & mean & median
vmean = santerne.groupby('nature')['FPP'].mean()
vmedian = santerne.groupby('nature')['FPP'].median()
count = santerne.groupby('nature')['FPP'].count()
vmean['huh'] = vmean['?']
vmedian['huh'] = vmedian['?']
count['huh'] = count['?']

outfile = open('document/table_santerne.tex', 'w')

outfile.write(r'\begin{deluxetable}{cccc}' + '\n')
outfile.write(r'\tabletypesize{\scriptsize}' + '\n')
outfile.write(r'\tablecaption{\vespa--calculated FPPs of the \\Santerne (2015) RV sample' + '\n')
outfile.write(r'  \tablabel{santerne}}' + '\n')
outfile.write(r'\tablehead{\colhead{RV-based nature} & \colhead{Number} & \colhead{mean FPP} & \colhead{median FPP}}' + '\n')
outfile.write(r'\startdata' + '\n')
outfile.write('Planet & {2.planet} & {0.planet:.2g} & {1.planet:.2g} \\\\ \n'.format(vmean,vmedian,count))
outfile.write('Brown dwarf & {2.BD} & {0.BD:.2g} & {1.BD:.2g} \\\\ \n'.format(vmean,vmedian,count))
outfile.write('Eclipsing binary (EB) & {2.EB} & {0.EB:.2g} & {1.EB:.2g} \\\\ \n'.format(vmean,vmedian,count))
outfile.write('Contaminating EB & {2.CEB} & {0.CEB:.2g} & {1.CEB:.2g} \\\\ \n'.format(vmean,vmedian,count))
outfile.write('Undetermined & {2.huh} & {0.huh:.2g} & {1.huh:.2g} \n'.format(vmean,vmedian,count))
outfile.write(r'\enddata' + '\n')
outfile.write(r'\end{deluxetable}' + '\n')
outfile.close()

santerne['FPP'].mean()

santerne.to_csv('santerne_sample_with_fpp.csv')

santerne.index = santerne.koi

santerne.ix['K00614.01']

santerne.query('nature=="planet" and FPP > 0.5')

fpp_all.ix['K00368.01']

import koiutils as ku
n_cands = []
for k in fpp_all.index:
    try:
        n_cands.append(ku.get_ncands(k))
    except:
        n_cands.append(np.nan)
        
fpp_all['n_cands'] = n_cands

fpp_all.query('n_cands == 1 and disposition=="CANDIDATE"')['FPP'].describe()



