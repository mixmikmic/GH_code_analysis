data = ["Spark is great for big data.", 
        "Big data cannot fit on one computer.",
        "Spark can be installed on a cluster of computers."
       ]

rdd = sc.parallelize(data)

rdd.take(1)

from time import time
def timeit(method):

    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()

        print "{} {}".format(method.__name__, te-ts)
        return result

    return timed

@timeit
def col_rdd(rdd):
    print rdd.collect()

@timeit
def sort_rdd(rdd):
    rdd.sortBy(lambda x: x[0]).take(1)

from time import time

t0 = time()
rdd_fm = rdd.flatMap(lambda text: text.split())
rdd_tup = rdd_fm.map(lambda word: (word.strip(".,-;?").lower(),1))
print "Time for completion, step 1:", time() - t0

col_rdd(rdd_fm)
sort_rdd(rdd_fm)

rdd_tup.cache()
terms = rdd_tup.reduceByKey(lambda a, b: a+b)            .sortBy(lambda x: x[1], ascending=False)            .take(15)
            
print terms



