get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
fpp_all = pd.read_csv('fpp_final_table.csv', index_col=0)
fpp_all.count()

import matplotlib.pyplot as plt

pem = fpp_all.ephem_match 
other_fp = ((fpp_all.disposition == 'FALSE POSITIVE') & 
            (fpp_all.not_transitlike | fpp_all.centroid_offset | 
                                     fpp_all.significant_secondary))
score_ok = fpp_all.pos_prob_score > 0.3
pos_ok = fpp_all.prob_ontarget > 0.99
ok = fpp_all.L_tot > 1e-3
lo_fpp = fpp_all.FPP < 0.01
plt.hist(fpp_all[pem & pos_ok].pos_prob_score)
print(sum(pem),sum(pem & pos_ok), sum(pem & pos_ok & score_ok))
print(sum(pem & pos_ok & score_ok & ok), sum(pem & pos_ok & score_ok & ok & lo_fpp))

print(pem & ~other_fp).sum()
print(sum(pem & pos_ok & score_ok & ~other_fp & ok), sum(pem & ~other_fp & pos_ok & score_ok & ok & lo_fpp))



sum(pem & (fpp_all.prob_ontarget==0))

q = 'pos_prob_score > 0.3 and L_tot > 1e-3 and prob_ontarget > 0.99 and ephem_match==1'
q2 = 'ephem_match==1'
print len(fpp_all.query(q))
print len(fpp_all.query(q2))
fpp_all.query(q)[['FPP','pos_prob_score','period','L_tot']]



