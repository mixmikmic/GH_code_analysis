import numpy as np
from dampingid import wt_damping_id
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

A=1
w=20 #[Hz]
zeta=0.02
t=np.linspace(0,2,600)

n=100 #number of samples to be generated. 
vSNR=np.zeros(n)
errcwt=np.zeros(n)
errswt=np.zeros(n)
errswt_avg=np.zeros(n)
errswt_prop=np.zeros(n)

for i,B in enumerate(np.logspace(-0.9,0.7,n)):

    xsig = A*np.sin(w*2*np.pi*t)*np.exp(-2*np.pi*w*zeta*t) #generated signal
    xnoise = B*(np.random.rand(len(t))-0.5) #generated noise
    varsig=np.std(xsig)**2 # variance of signal
    varnoise=np.std(xnoise)**2 #variance of noise
    vSNR[i] = 10*np.log10(varsig/varnoise) # signal to noise ratio

    x=xsig+xnoise 
    WT = wt_damping_id(x,t,np.linspace(15,25,100),5)

    errcwt[i] = abs(0.02-WT.ident('cwt',50)[1])/0.02*100 #error using the CWT
    errswt[i] = abs(0.02-WT.ident('swt',50)[1])/0.02*100 #error using the SWT
    errswt_avg[i] = abs(0.02-WT.ident('swt_avg',50)[1])/0.02*100 #error using the averaged SWT
    errswt_prop[i] = abs(0.02-WT.ident('swt_prop',50)[1])/0.02*100 #error using the proportional SWT

_ = vSNR.argsort()
svSNR = vSNR[_] #sorting the SNR ascending
serrcwt = errcwt[_] #sorting accordingly to SNR
serrswt = errswt[_] #sorting accordingly to SNR
serrswt_avg = errswt_avg[_] #sorting accordingly to SNR
serrswt_prop = errswt_prop[_] #sorting accordingly to SNR

plt.figure(num=None, figsize=(5, 3), dpi=300, facecolor='w', edgecolor='k')
plt.plot(svSNR[15:-15],np.convolve(serrswt,np.ones((31))/31., mode='valid'),'y-*',ms=10,markevery=10,lw=2,label='SWT')
plt.plot(svSNR[15:-15],np.convolve(serrcwt,np.ones((31))/31., mode='valid'),'ro-',ms=7,markevery=10,lw=2,label='CWT')
plt.plot(svSNR[15:-15],np.convolve(serrswt_avg,np.ones((31))/31., mode='valid'),'g-^',ms=7,markevery=10,lw=2,label='averaged SWT')
plt.plot(svSNR[15:-15],np.convolve(serrswt_prop,np.ones((31))/31., mode='valid'),'b-s',ms=7,markevery=10,lw=2,label='proportional SWT')
plt.gca().invert_xaxis()
plt.ylim(0,80)
plt.xlabel('SNR [dB]')
plt.ylabel('error [%]')
plt.legend(loc=2)
plt.title('Error of identified damping ratio')



