from pyspark import SparkContext
sc = SparkContext(master="local[4]")
sc

# sc.stop() #commented out so that you don't stop your context by mistake

A=sc.parallelize(range(3))
A

L=A.collect()
print type(L)
print L

A.map(lambda x: x*x).collect()

A.reduce(lambda x,y:x+y)

words=['this','is','the','best','mac','ever']
wordRDD=sc.parallelize(words)
wordRDD.reduce(lambda w,v: w if len(w)<len(v) else v)

B=sc.parallelize([1,3,5,2])
B.reduce(lambda x,y: x-y)

A.reduce(lambda x,y: x+y)

def largerThan(x,y):
    if len(x)>len(y): return x
    elif len(y)>len(x): return y
    else:  #lengths are equal, compare lexicographically
        if x>y: 
            return x
        else: 
            return y
        
wordRDD.reduce(largerThan)

