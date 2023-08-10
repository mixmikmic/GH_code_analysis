from time import time
from pyspark import SparkContext
for j in range(1,10):
    sc = SparkContext(master="local[%d]"%(j))
    t0=time()
    for i in range(10):
        sc.parallelize([1,2]*1000000).reduce(lambda x,y:x+y)
    print "%2d executors, time=%4.3f"%(j,time()-t0)
    sc.stop()

sc.stop()



