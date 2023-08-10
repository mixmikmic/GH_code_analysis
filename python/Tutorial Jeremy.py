rdd=sc.parallelize([1,2,3])

rdd

rdd.count()

rdd2=rdd.map(lambda x: x+1)

tryseries=rdd.toSeries()

rdd.collect()

_=1

a=2

import numpy as np

def func(x):
    return x+1

files=['a.png','b.png']
sc.paralllize(files).map(load)

rdd = sc.parallelize([(1,'a'),(2,'g')])

rdd.keys().reduce(lambda x,y:x+y)



