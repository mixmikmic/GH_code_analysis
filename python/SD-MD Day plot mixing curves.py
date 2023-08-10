import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

Mr_Ms_SD = 0.380
Mr_Ms_MD = 0.019

Hc_SD = 396
Hc_MD = 32

Hcr_SD = 565
Hcr_MD = 160

chi_SD = 0.465
chi_MD = 0.288

chi_r_SD = 0.326
chi_r_MD = 0.058

frac_SD_model1 = []
frac_MD_model1 = []
Mrs_Ms_model1 = []
Hcr_Hc_model1 = []

for frac in np.arange(0,1,.01):
    frac_SD = frac
    frac_MD = 1-frac
    Mrs_Ms = frac_SD*Mr_Ms_SD + frac_MD*Mr_Ms_MD
    Hc = (frac_SD*chi_SD*Hc_SD + frac_MD*chi_MD*Hc_MD)/(frac_SD*chi_SD + frac_MD*chi_MD)
    Hcr = (frac_SD*chi_r_SD*Hcr_SD + frac_MD*chi_r_MD*Hcr_MD)/(frac_SD*chi_r_SD + frac_MD*chi_r_MD)
    
    frac_SD_model1.append(frac_SD)
    frac_MD_model1.append(frac_MD)
    Mrs_Ms_model1.append(Mrs_Ms)
    Hcr_Hc_model1.append(Hcr/Hc)

Mr_Ms_SD = 0.5
Mr_Ms_MD = 0.019

Hc_SD = 400.0
Hc_MD = 43.0

Hcr_SD = 500.0
Hcr_MD = 230.0

chi_SD = 0.600
chi_MD = 0.209

chi_r_SD = 0.480
chi_r_MD = 0.039

frac_SD_model2 = []
frac_MD_model2 = []
Mrs_Ms_model2 = []
Hcr_Hc_model2 = []

for frac in np.arange(0,1,.01):
    frac_SD = frac
    frac_MD = 1-frac
    Mrs_Ms = frac_SD*Mr_Ms_SD + frac_MD*Mr_Ms_MD
    Hc = (frac_SD*chi_SD*Hc_SD + frac_MD*chi_MD*Hc_MD)/(frac_SD*chi_SD + frac_MD*chi_MD)
    Hcr = (frac_SD*chi_r_SD*Hcr_SD + frac_MD*chi_r_MD*Hcr_MD)/(frac_SD*chi_r_SD + frac_MD*chi_r_MD)
    
    frac_SD_model2.append(frac_SD)
    frac_MD_model2.append(frac_MD)
    Mrs_Ms_model2.append(Mrs_Ms)
    Hcr_Hc_model2.append(Hcr/Hc)

plt.plot(Hcr_Hc_model1,Mrs_Ms_model1,'b',label='SD-MD model 1')
plt.plot(Hcr_Hc_model2,Mrs_Ms_model2,'k',label='SD-MD model 2')
plt.scatter(Hcr_SD/Hc_SD,Mr_Ms_SD,c='r',label='SD2 end member')
plt.scatter(Hcr_MD/Hc_MD,Mr_Ms_MD,c='g',label='MD2 end member')
plt.xlim(1,7)
plt.ylim(0,0.53)
plt.ylabel('M$_{r}$/M$_{s}$')
plt.xlabel('B$_{cr}$/B$_{c}$')
plt.legend()
plt.show()

plt.loglog(Hcr_Hc_model1,Mrs_Ms_model1,'b',label='SD-MD model 1')
plt.loglog(Hcr_Hc_model2,Mrs_Ms_model2,'k',label='SD-MD model 2')
plt.scatter(Hcr_SD/Hc_SD,Mr_Ms_SD,c='r',label='SD2 end member')
plt.scatter(Hcr_MD/Hc_MD,Mr_Ms_MD,c='g',label='MD2 end member')
plt.xlim(1,20)
plt.ylim(0.01,1)
plt.ylabel('M$_{r}$/M$_{s}$')
plt.xlabel('B$_{cr}$/B$_{c}$')
plt.legend()
plt.show()



