get_ipython().magic('pylab inline')

N = 1  # < ---- Increase this number to any integer greater than 1
M = 1000
A = 1
ra = arange(M) * 100./M
signal = zeros((N,M))
seed(8)
signal1 = exp(-((ra-50.)/10.)**2)
signal2 = exp(-((ra-20.)/10.)**2)
signal3 = exp(-((ra-25.)/10.)**2)
signal4 = exp(-((ra-80.)/3)**2)
signal5 = 0.5*exp(-((ra-10.)/3.)**2)
for i in range(N):
    signal6 = A*randn(M)
    signal7 = signal1+signal2+signal3+signal4+signal5+signal6
    signal[i,:] = signal7
plot(ra,signal.mean(axis=0))

