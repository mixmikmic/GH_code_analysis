import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
#matplotlib.use('nbagg')
import numpy as np
get_ipython().magic('matplotlib inline')
import sygma as s
reload(s)

#import mpld3
#mpld3.enable_notebook()

k_N=1e11*0.35/ (1**-0.35 - 30**-0.35) #(I)
N_tot=k_N/1.35 * (1**-1.35 - 30**-1.35) #(II)
Yield_tot=0.1*N_tot
print Yield_tot/1e11

s1=s.sygma(iolevel=0,mgal=1e11,dt=1e7,tend=1.3e10,imf_type='salpeter',alphaimf=2.35,imf_bdys=[1,30],iniZ=0.02,hardsetZ=0.0001,           table='yield_tables/isotope_yield_table_h1.txt',sn1a_on=False, sn1a_table='yield_tables/sn1a_h1.txt',            iniabu_table='yield_tables/iniabu/iniab1.0E-04GN93_alpha_h1.ppn')
s2=s.sygma(iolevel=0,mgal=1e11,dt=1e7,tend=1.3e10,imf_type='input',imf_bdys=[1,30],iniZ=0.02,hardsetZ=0.0001,           table='yield_tables/isotope_yield_table_h1.txt',sn1a_on=False, sn1a_table='yield_tables/sn1a_h1.txt',            iniabu_table='yield_tables/iniabu/iniab1.0E-04GN93_alpha_h1.ppn')

s1.plot_totmasses(fig=3,mass='gas',source='all',norm='no',color='k',marker='x',shape=':',markevery=20,label='s')
s2.plot_totmasses(fig=3,mass='gas',source='all',norm='no',color='g',marker='o',shape='--',markevery=20,label='-2.35 custom')
plt.show()

s1=s.sygma(iolevel=0,mgal=1e11,dt=1e7,tend=1.3e10,imf_type='alphaimf',alphaimf=1.5,imf_bdys=[1,30],iniZ=0.02,hardsetZ=0.0001,           table='yield_tables/isotope_yield_table_h1.txt',sn1a_on=False, sn1a_table='yield_tables/sn1a_h1.txt',            iniabu_table='yield_tables/iniabu/iniab1.0E-04GN93_alpha_h1.ppn')
s2=s.sygma(iolevel=0,mgal=1e11,dt=1e7,tend=1.3e10,imf_type='input',imf_bdys=[1,30],iniZ=0.02,hardsetZ=0.0001,           table='yield_tables/isotope_yield_table_h1.txt',sn1a_on=False, sn1a_table='yield_tables/sn1a_h1.txt',            iniabu_table='yield_tables/iniabu/iniab1.0E-04GN93_alpha_h1.ppn')

s1=s.sygma(iolevel=0,mgal=1e11,dt=1e7,tend=1.3e10,imf_type='kroupa',alphaimf=1.5,imf_bdys=[1,30],iniZ=0.02,hardsetZ=0.0001,           table='yield_tables/isotope_yield_table_h1.txt',sn1a_on=False, sn1a_table='yield_tables/sn1a_h1.txt',            iniabu_table='yield_tables/iniabu/iniab1.0E-04GN93_alpha_h1.ppn')
s2=s.sygma(iolevel=0,mgal=1e11,dt=1e7,tend=1.3e10,imf_type='input',imf_bdys=[1,30],iniZ=0.02,hardsetZ=0.0001,           table='yield_tables/isotope_yield_table_h1.txt',sn1a_on=False, sn1a_table='yield_tables/sn1a_h1.txt',            iniabu_table='yield_tables/iniabu/iniab1.0E-04GN93_alpha_h1.ppn')

s1.plot_totmasses(fig=2,mass='gas',source='all',norm='no',color='k',marker='x',shape=':',markevery=20,label='default kroupa')
s2.plot_totmasses(fig=2,mass='gas',source='all',norm='no',color='g',marker='o',shape='--',markevery=20,label='kroupa custom')
plt.show()



