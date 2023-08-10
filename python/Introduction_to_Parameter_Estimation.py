from sie import *

dist=beta(h=1,N=3)
distplot(dist,xlim=[0,1],show_quartiles=False)

dist.median()

credible_interval(dist)

dist=beta(h=1,N=4)
distplot(dist,xlim=[0,1])

credible_interval(dist)



