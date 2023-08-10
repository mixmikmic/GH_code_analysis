from sie import *

x=[5.1, 4.9, 4.7, 4.9, 5.0]
sigma=0.5

mu=sample_mean(x)
N=len(x)

dist=normal(mu,sigma/sqrt(N))
distplot(dist)

credible_interval(dist)

mu=sample_mean(x)
s=sample_deviation(x)
print mu,s

dist=tdist(N-1,mu,s/sqrt(N))

distplot(dist,xlim=[4.6,5.4])

credible_interval(dist)









