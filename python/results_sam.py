import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import kendalltau, spearmanr, pearsonr
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn import decomposition
import pandas as pd
import seaborn as sns
import time
import math
get_ipython().magic('matplotlib inline')

sns.set_style("white")
sns.set_style("ticks")

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

get_ipython().magic("run 'helper_sam.ipynb'")

get_ipython().system('wget --http-user=<username> --http-passwd=<password> -O data.csv "http://gavo.mpa-garching.mpg.de/MyMillennium?action=doQuery&SQL=SELECT a.np, a.vMax, a.halfmassRadius, a.vDisp, a.spinX, a.spinY,blah.r_mean200,blah.r_crit200, blah.m_Crit200,a.spinZ,g.rvir,g.centralMvir,g.vvir,g1.rvir,g1.centralMvir,g1.vvir,a.m_mean200,a.m_crit200,a.m_tophat,a.peakMassMcrit200,a.peakMassVmax,b.np,b.vMax,b.vDisp,b.m_crit200,b.spinZ,b.lastCentralVmax,b.m_tophat,c.np,c.vMax,c.vDisp,c.m_tophat,c.m_crit200,c.spinZ,c.lastCentralVmax,d.np,d.vMax,d.vDisp,d.m_tophat,d.m_crit200,d.spinZ,d.lastCentralVmax,e.np,e.vMax,e.vDisp,e.m_crit200,e.spinZ,f.np,f.vMax,f.vDisp,f.m_crit200,f.spinZ,h.np,h.vMax,h.vDisp,h.m_crit200,h.spinZ,i.np,i.vMax,i.vDisp,i.m_crit200,i.spinZ,j.np,j.vMax,j.vDisp,j.m_crit200,j.spinZ,k.np,k.vMax,k.vDisp,k.m_crit200,k.spinZ,l.np,l.vMax,l.vDisp,l.m_crit200,l.spinZ,m.np,m.vMax,m.vDisp,m.m_crit200,m.spinZ,n.np,n.vMax,n.vDisp,n.m_crit200,n.spinZ,o.np,o.vMax,o.vDisp,o.m_crit200,o.spinZ,p.np,p.vMax,p.vDisp,p.m_crit200,p.spinZ,q.np,q.vMax,q.vDisp,q.m_crit200,q.spinZ,r.np,r.vMax,r.vDisp,r.m_crit200,s.np,s.vMax,s.vDisp,s.m_crit200,t.np,t.vMax,t.vDisp,t.m_crit200,u.np,u.vMax,u.vDisp,u.m_crit200,v.np,v.vMax,v.vDisp,v.m_crit200,w.np,w.vMax,w.vDisp,w.m_crit200,x.np,x.vMax,x.vDisp,x.m_crit200,y.vMax,y.vDisp,y.m_crit200,z.vMax,z.vDisp,z.m_crit200,ab.vMax,ab.vDisp,ab.m_crit200,bc.vMax,bc.vDisp,bc.m_crit200,cd.vMax,cd.vDisp,cd.m_crit200,de.vMax,de.vDisp,de.m_crit200,ef.vMax,ef.vDisp,ef.m_crit200,fg.vMax,fg.vDisp,fg.m_crit200,gh.vMax,gh.vDisp,gh.m_crit200,hi.vMax,hi.vDisp,hi.m_crit200,ij.vMax,ij.vDisp,ij.m_crit200,jk.vMax,jk.vDisp,jk.m_crit200,kl.vMax,kl.vDisp,kl.m_crit200,lm.vMax,lm.vDisp,lm.m_crit200,mn.vMax,mn.vDisp,mn.m_crit200,no.vMax,no.vDisp,no.m_crit200,op.vMax,op.vDisp,op.m_crit200,pq.vMax,pq.vDisp,pq.m_crit200,qr.vMax,qr.vDisp,qr.m_crit200,rs.vMax,rs.vDisp,rs.m_crit200,st.vMax,st.vDisp,st.m_crit200,g1.coolingradius,g1.hotGas,g.stellarMass,g.coldGas,g.bulgeMass,g.hotGas,g.coolingRadius,g.blackHoleMass FROM MPAHaloTrees..MR as a INNER JOIN MField..FofSubHalo as mfs ON mfs.subhaloId = a.subhaloFileId INNER JOIN MField..FOF as blah ON blah.fofId = mfs.fofId INNER JOIN Guo2010a..MR as g on g.haloID = a.haloId INNER JOIN MPAHaloTrees..MR as b on b.haloId = a.firstProgenitorId INNER JOIN Guo2010a..MR as g1 on g1.haloID = b.haloId INNER JOIN MPAHaloTrees..MR as c on c.haloId = b.firstProgenitorId INNER JOIN MPAHaloTrees..MR as d on d.haloId = c.firstProgenitorId INNER JOIN MPAHaloTrees..MR as e on e.haloId = d.firstProgenitorId INNER JOIN MPAHaloTrees..MR as f on f.haloId = e.firstProgenitorId INNER JOIN MPAHaloTrees..MR as h on h.haloId = f.firstProgenitorId INNER JOIN MPAHaloTrees..MR as i on i.haloId = h.firstProgenitorId INNER JOIN MPAHaloTrees..MR as j on j.haloId = i.firstProgenitorId INNER JOIN MPAHaloTrees..MR as k on k.haloId = j.firstProgenitorId INNER JOIN MPAHaloTrees..MR as l on l.haloId = k.firstProgenitorId INNER JOIN MPAHaloTrees..MR as m on m.haloId = l.firstProgenitorId INNER JOIN MPAHaloTrees..MR as n on n.haloId = m.firstProgenitorId INNER JOIN MPAHaloTrees..MR as o on o.haloId = n.firstProgenitorId INNER JOIN MPAHaloTrees..MR as p on p.haloId = o.firstProgenitorId INNER JOIN MPAHaloTrees..MR as q on q.haloId = p.firstProgenitorId INNER JOIN MPAHaloTrees..MR as r on r.haloId = q.firstProgenitorId INNER JOIN MPAHaloTrees..MR as s on s.haloId = r.firstProgenitorId INNER JOIN MPAHaloTrees..MR as t on t.haloId = s.firstProgenitorId INNER JOIN MPAHaloTrees..MR as u on u.haloId = t.firstProgenitorId INNER JOIN MPAHaloTrees..MR as v on v.haloId = u.firstProgenitorId INNER JOIN MPAHaloTrees..MR as w on w.haloId = v.firstProgenitorId INNER JOIN MPAHaloTrees..MR as x on x.haloId = w.firstProgenitorId INNER JOIN MPAHaloTrees..MR as y on y.haloId = x.firstProgenitorId INNER JOIN MPAHaloTrees..MR as z on z.haloId = y.firstProgenitorId INNER JOIN MPAHaloTrees..MR as ab on ab.haloId = z.firstProgenitorId INNER JOIN MPAHaloTrees..MR as bc on bc.haloId = ab.firstProgenitorId INNER JOIN MPAHaloTrees..MR as cd on cd.haloId = bc.firstProgenitorId INNER JOIN MPAHaloTrees..MR as de on de.haloId = cd.firstProgenitorId INNER JOIN MPAHaloTrees..MR as ef on ef.haloId = de.firstProgenitorId INNER JOIN MPAHaloTrees..MR as fg on fg.haloId = ef.firstProgenitorId INNER JOIN MPAHaloTrees..MR as gh on gh.haloId = fg.firstProgenitorId INNER JOIN MPAHaloTrees..MR as hi on hi.haloId = gh.firstProgenitorId INNER JOIN MPAHaloTrees..MR as ij on ij.haloId = hi.firstProgenitorId INNER JOIN MPAHaloTrees..MR as jk on jk.haloId = ij.firstProgenitorId INNER JOIN MPAHaloTrees..MR as kl on kl.haloId = jk.firstProgenitorId INNER JOIN MPAHaloTrees..MR as lm on lm.haloId = kl.firstProgenitorId INNER JOIN MPAHaloTrees..MR as mn on mn.haloId = lm.firstProgenitorId INNER JOIN MPAHaloTrees..MR as no on no.haloId = mn.firstProgenitorId INNER JOIN MPAHaloTrees..MR as op on op.haloId = no.firstProgenitorId INNER JOIN MPAHaloTrees..MR as pq on pq.haloId = op.firstProgenitorId INNER JOIN MPAHaloTrees..MR as qr on qr.haloId = pq.firstProgenitorId INNER JOIN MPAHaloTrees..MR as rs on rs.haloId = qr.firstProgenitorId INNER JOIN MPAHaloTrees..MR as st on st.haloId = rs.firstProgenitorId WHERE (a.np > 1162.67) AND (g.type=0) AND (a.snapnum=63) AND (g1.type=0)"')

df = pd.read_csv('/Users/harshilkamdar/Desktop/paper/data.csv') #load in the data
print(df.shape) 

Q = df.values
M = Q[:,195:201] #mass; need to do this in an iffy way (i.e. not using pandas) because there are duplicate labels

means = np.mean(M, axis=0) 
stds = np.std(M, axis=0)
cutoffs = means + 30*stds #cutoffs to remove extreme outliers

df = df[df.stellarMass < cutoffs[0]][df.coldGas < cutoffs[1]][df.bulgeMass < cutoffs[2]][df.hotGas < cutoffs[3]][df.blackHoleMass < cutoffs[5]] #god bless pandas
print df.shape #just to see how many entries we lost; it turns out to be only 48 out of 350k+ 

Q = df.values #same here; this is really not recommended since
H = Q[:,0:193] #halo inputs; no baryonic quantities are included here
M = Q[:,195:201] #galaxy masses

training_size = 0.35 

H_train, H_test, M_train, M_test = cross_validation.train_test_split(H, M, train_size=training_size, random_state=23) #the random state is chosen for consistency across different runs

HB = np.c_[H, Q[:,193:195], M[:,3], M[:,4]] #halo inputs with cooling radius and hot gas from the last two snapshots
C = np.c_[M[:,1]] #just the cold gas mass

N_train, N_test, C_train, C_test = cross_validation.train_test_split(HB, C, train_size=training_size, random_state=23)

base_mse_st = mse(M_test[:,0], np.mean(M_train[:,0])) 
base_mse_co = mse(M_test[:,1], np.mean(M_train[:,1])) 
base_mse_bu = mse(M_test[:,2], np.mean(M_train[:,2]))
base_mse_ho = mse(M_test[:,3], np.mean(M_train[:,3]))
base_mse_rc = mse(M_test[:,4], np.mean(M_train[:,4]))
base_mse_bh = mse(M_test[:,5], np.mean(M_train[:,5]))

base_mse = np.c_[base_mse_st, base_mse_co, base_mse_bu, base_mse_ho, base_mse_rc, base_mse_bh] 

print('Base MSE for stellar mass is: %f, cold gas mass is: %f, stellar mass in the bulge is: %f, hot gas mass is: %f, cooling radius is: %f and black hole mass is: %f' % (base_mse_st, base_mse_co, base_mse_bu, base_mse_ho, base_mse_rc, base_mse_bh))

titles = ['$M_{\star}$', '$M_{cold}$', '$M_{bulge}$', '$M_{hot}$', '$R_{cooling}$', '$M_{BH}$']

del df
del H, M, HB, C #delete everything from memory because we weren't exactly memory conscious. oops

sns.palplot(sns.cubehelix_palette(18, start=3, rot=0.1, dark=0, light=0.99))
cold_cmap = sns.cubehelix_palette(18, start=3, rot=0.1, dark=0, light=0.99, as_cmap=True)

'''
param_dist = {<some parameter space to search over>}
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid=param_dist)
grid_search.fit(B_train, M_train)
report(grid_search.grid_scores_)

Commented out because a full grid search will take too long with such a big multidimensional data set. The report function (in helper.ipynb) will print the 3 best parameter combinations. Included here for completeness. 
'''

knn = KNeighborsRegressor(60, weights='distance') #knn is cool, i guess
M_pred_knn = knn.fit(H_train, M_train).predict(H_test)
mse_knn = mse(M_test, M_pred_knn)
del knn 

print('MSE using kNN for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (mse_knn[0], mse_knn[1],  mse_knn[2], mse_knn[3], mse_knn[4], mse_knn[5]))

base_mse = np.ravel(base_mse)

factors = base_mse/mse_knn

print('Factor reduction (MSE_b/MSE) for kNN for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (factors[0], factors[1],  factors[2], factors[3], factors[4], factors[5])) #TODO: prettier way to print this stuff
print('R^2 score for kNN for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (r2_score(M_test[:,0], M_pred_knn[:,0]), r2_score(M_test[:,1], M_pred_knn[:,1]),  r2_score(M_test[:,2], M_pred_knn[:,2]), r2_score(M_test[:,3], M_pred_knn[:,3]), r2_score(M_test[:,4], M_pred_knn[:,4]), r2_score(M_test[:,5], M_pred_knn[:,5])))
print('Pearson correlation for kNN for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f' % (pearsonr(M_test[:,0], M_pred_knn[:,0])[0], pearsonr(M_test[:,1], M_pred_knn[:,1])[0],  pearsonr(M_test[:,2], M_pred_knn[:,2])[0], pearsonr(M_test[:,3], M_pred_knn[:,3])[0], pearsonr(M_test[:,4], M_pred_knn[:,4])[0], pearsonr(M_test[:,5], M_pred_knn[:,5])[0]))

[genplots_M(np.c_[M_test[:,i], M_pred_knn[:,i]], mse_knn[i]) for i in [x for x in xrange(0,6) if x != 4]]
genplots_M(np.c_[M_test[:,4], M_pred_knn[:,4]], mse_knn[4], plot_type='R') 

'''
param_dist = {<some parameter space to search over>}
dtree_r = DecisionTreeRegressor(random_state=0)
grid_search = GridSearchCV(dtree_r, param_grid=param_dist)
grid_search.fit(B_train, M_train)
report(grid_search.grid_scores_)

Commented out because a full grid search will take too long with such a big multidimensional data set. The report function (in helper.ipynb) will print the 3 best parameter combinations. Included here for completeness. 
'''
dtree_r = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=3) #treez
M_pred_dtree = dtree_r.fit(H_train, M_train).predict(H_test)
mse_dtree = mse(M_test, M_pred_dtree)
del dtree_r

print('MSE using Decision Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (mse_dtree[0], mse_dtree[1],  mse_dtree[2], mse_dtree[3], mse_dtree[4], mse_dtree[5]))

factors = base_mse/mse_dtree

print('Factor reduction (MSE_b/MSE) using Decision Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (factors[0], factors[1],  factors[2], factors[3], factors[4], factors[5])) #TODO: prettier way to print this stuff
print('R^2 score using Decision Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (r2_score(M_test[:,0], M_pred_dtree[:,0]), r2_score(M_test[:,1], M_pred_dtree[:,1]),  r2_score(M_test[:,2], M_pred_dtree[:,2]), r2_score(M_test[:,3], M_pred_dtree[:,3]), r2_score(M_test[:,4], M_pred_dtree[:,4]), r2_score(M_test[:,5], M_pred_dtree[:,5])))
print('Pearson correlation using Decision Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f' % (pearsonr(M_test[:,0], M_pred_dtree[:,0])[0], pearsonr(M_test[:,1], M_pred_dtree[:,1])[0],  pearsonr(M_test[:,2], M_pred_dtree[:,2])[0], pearsonr(M_test[:,3], M_pred_dtree[:,3])[0], pearsonr(M_test[:,4], M_pred_dtree[:,4])[0], pearsonr(M_test[:,5], M_pred_dtree[:,5])[0]))

[genplots_M(np.c_[M_test[:,i], M_pred_dtree[:,i]], mse_dtree[i]) for i in [x for x in xrange(0,6) if x != 4]]
genplots_M(np.c_[M_test[:,4], M_pred_dtree[:,4]], mse_dtree[4], plot_type='R') 

'''
param_dist = {<some paramter grid>}
rf = RandomForestRegressor(random_state=0)
grid_search = GridSearchCV(rf, param_grid=param_dist)
grid_search.fit(B_train, M_train)

report(grid_search.grid_scores_)

Commented out because a full grid search will take too long with such a big multidimensional data set. The report function (in helper.ipynb) will print the 3 best parameter combinations. Included here for completeness. 
'''
#rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, min_samples_split=5, max_features=100)
#rf.fit(H_train, M_train)
#M_pred_rf = rf.predict(H_test)

mse_rf = mse(M_test, M_pred_rf)

print('MSE using Random Forests for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (mse_rf[0], mse_rf[1], mse_rf[2], mse_rf[3], mse_rf[4], mse_rf[5]))

base_mse = np.s_[base_mse_st, base_mse_co, base_mse_bu, base_mse_ho, base_mse_rc, base_mse_bh] 
print base_mse

factors = base_mse/mse_rf

print('Factor reduction (MSE_b/MSE) for Random Forests for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (factors[0], factors[1],  factors[2], factors[3], factors[4], factors[5])) #TODO: prettier way to print this stuff
print('R^2 score using Random Forests for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (r2_score(M_test[:,0], M_pred_rf[:,0]), r2_score(M_test[:,1], M_pred_rf[:,1]),  r2_score(M_test[:,2], M_pred_rf[:,2]), r2_score(M_test[:,3], M_pred_rf[:,3]), r2_score(M_test[:,4], M_pred_rf[:,4]), r2_score(M_test[:,5], M_pred_rf[:,5])))
print('Pearson correlation using Random Forests for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f' % (pearsonr(M_test[:,0], M_pred_rf[:,0])[0], pearsonr(M_test[:,1], M_pred_rf[:,1])[0],  pearsonr(M_test[:,2], M_pred_rf[:,2])[0], pearsonr(M_test[:,3], M_pred_rf[:,3])[0], pearsonr(M_test[:,4], M_pred_rf[:,4])[0], pearsonr(M_test[:,5], M_pred_rf[:,5])[0]))

[genplots_M(np.c_[M_test[:,i], M_pred_rf[:,i]], mse_rf[i]) for i in [x for x in xrange(0,6) if x != 4]]
genplots_M(np.c_[M_test[:,4], M_pred_rf[:,4]], mse_rf[4], plot_type='R') 

plot_smhm(M_test[:,0], M_pred_rf[:,0], H_test[:,0])
plot_bhbulge(M_test[:,5], M_pred_rf[:,5], M_test[:,2], M_pred_rf[:,2])
plot_coldgasfrac(M_test[:,1], M_pred_rf[:,1], M_test[:,0], M_pred_rf[:,0])

C_train = np.ravel(C_train)
C_test = np.ravel(C_test)

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, min_samples_split=5)
rf.fit(N_train, C_train)
C_pred_rf = rf.predict(N_test)

mse_rf_mod = mse(C_test,C_pred_rf)

print('MSE using kNN for predicting the cold gas mass including the baryonic inputs is %f' % mse_rf_mod)
print('R^2 score using Random Forests for predicting the cold gas mass including the baryonic inputs is %f' %  r2_score(C_test[:], C_pred_rf[:]))
print('Pearson correlation using Random Forests for predicting the cold gas mass including the baryonic inputs is %f' % pearsonr(C_test[:], C_pred_rf[:])[0])

genplots_M(np.c_[C_test[:],C_pred_rf[:]], mse_rf_mod) 

'''
param_dist = {"n_estimators":[43,46,50],
              "max_depth":[10,12,14]
              }
etree = ExtraTreesRegressor()
grid_search = GridSearchCV(etree, param_grid=param_dist)
grid_search.fit(B_train, M_train)

report(grid_search.grid_scores_)
'''
etree = ExtraTreesRegressor(n_estimators=700, min_samples_split=5, n_jobs=-1)
etree.fit(H_train, M_train)
M_pred_etree = etree.predict(H_test)

mse_etree = mse(M_test, M_pred_etree)
del etree

print('MSE using Extra Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (mse_etree[0], mse_etree[1], mse_etree[2], mse_etree[3], mse_etree[4], mse_etree[5]))

factors = base_mse/mse_etree

print('Factor reduction (MSE_b/MSE) for Extra Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (factors[0], factors[1],  factors[2], factors[3], factors[4], factors[5])) #TODO: prettier way to print this stuff
print('R^2 score using Extra Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f\n' % (r2_score(M_test[:,0], M_pred_etree[:,0]), r2_score(M_test[:,1], M_pred_etree[:,1]),  r2_score(M_test[:,2], M_pred_etree[:,2]), r2_score(M_test[:,3], M_pred_etree[:,3]), r2_score(M_test[:,4], M_pred_etree[:,4]), r2_score(M_test[:,5], M_pred_etree[:,5])))
print('Pearson correlation using Extra Trees for predicting the stellar mass is %f, cold gas mass is %f, bulge mass is %f, hot gas mass is %f, cooling radius is %f and black hole mass is %f' % (pearsonr(M_test[:,0], M_pred_etree[:,0])[0], pearsonr(M_test[:,1], M_pred_etree[:,1])[0],  pearsonr(M_test[:,2], M_pred_etree[:,2])[0], pearsonr(M_test[:,3], M_pred_etree[:,3])[0], pearsonr(M_test[:,4], M_pred_etree[:,4])[0], pearsonr(M_test[:,5], M_pred_etree[:,5])[0]))

[genplots_M(np.c_[M_test[:,i], M_pred_etree[:,i]], mse_etree[i]) for i in [x for x in xrange(0,6) if x != 4]]
genplots_M(np.c_[M_test[:,4], M_pred_etree[:,4]], mse_etree[4], plot_type='R') 

plot_smhm(M_test[:,0], M_pred_etree[:,0], H_test[:,0])
plot_bhbulge(M_test[:,5], M_pred_etree[:,5], M_test[:,2], M_pred_etree[:,2])
plot_coldgasfrac(M_test[:,1], M_pred_etree[:,1], M_test[:,0], M_pred_etree[:,0])

C_train = np.ravel(C_train)
C_test = np.ravel(C_test)

etree = ExtraTreesRegressor(n_estimators=700, n_jobs=-1, min_samples_split=5)
etree.fit(N_train, C_train)
C_pred_etree = etree.predict(N_test)

mse_etree_mod = mse(C_test,C_pred_etree)

print('MSE using Extra Trees for predicting the cold gas mass including the baryonic inputs is %f' % mse_etree_mod)
print('R^2 score using Extra Trees for predicting the cold gas mass including the baryonic inputs is %f' %  r2_score(C_test[:], C_pred_etree[:]))
print('Pearson correlation using Extra Trees for predicting the cold gas mass including the baryonic inputs is %f' % pearsonr(C_test[:], C_pred_etree[:])[0])

genplots_M(np.c_[C_test[:],C_pred_rf[:]], mse_rf_mod)

'''H_train_fi = H_train[:,:5]
H_test_fi = H_test[:,:5]
H_train_fi = np.c_[H_train_fi, H_train[:,8]]
H_test_fi = np.c_[H_test_fi, H_test[:,8]]
print np.shape(H_train_fi)
print np.shape(H_test_fi)
rf = ExtraTreesRegressor(n_estimators=00, n_jobs=-1, min_samples_split=5)
rf.fit(H_train_fi, M_train)
M_pred_rf_fi = rf.predict(H_test_fi)
'''
from sklearn import datasets

boston = datasets.load_boston()
print type(boston.feature_names)
print(boston.feature_names[sorted_idx])

feature_importance = rf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center', alpha=0.7)
feature_names = np.array(('$Number$ $of$ $Particles$', '$V_{max}$', '$R_{half}$', '$V_{disp}$', '$Spin$', '$M_{crit,200}$'))
print(feature_names)
plt.yticks(pos, feature_names[sorted_idx])
plt.xlabel('$Relative$ $Importance$')
plt.savefig('features.pdf', bbox_inches='tight')
plt.show()



