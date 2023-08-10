# %load pyspark_init.py
"""
Load packages and create context objects...
"""
import os
import platform
import sys
if not 'sc' in vars():
    sys.path.append('/usr/hdp/2.4.2.0-258/spark/python')
    os.environ["SPARK_HOME"] = '/usr/hdp/2.4.2.0-258/spark'
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-csv_2.11:1.2.0 pyspark-shell'
    import py4j
    import pyspark
    from pyspark.context import SparkContext, SparkConf
    from pyspark.sql import SQLContext, HiveContext
    from pyspark.storagelevel import StorageLevel
    sc = SparkContext()
    import atexit
    atexit.register(lambda: sc.stop())
    print("""Welcome to
          ____              __
         / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
       /__ / .__/\_,_/_/ /_/\_\   version %s
          /_/
    """ % sc.version)
else:
    print("""Already running
          ____              __
         / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
       /__ / .__/\_,_/_/ /_/\_\   version %s
          /_/
    """ % sc.version)

if not 'sqlCtx' in vars():
    sqlCtx = SQLContext(sc)
print 'Spark Context available as `sc`'
print 'Spark SQL Context (%s) available as `sqlCtx`'%str(type(sqlCtx))
print "Monitor this application at http://arc.insight.gsu.edu:8088/proxy/"+sc.applicationId


review_rdd = sc.textFile('/Users/Peter/Downloads/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json').sample(False, 0.01, 42)

review_rdd.first()

rtrain_rdd, rtest_rdd = review_rdd.randomSplit([0.8, 0.2])

rtrain_rdd.count()



text =  "Mr Hoagie is an institution. Walking in, it does seem like a throwback to 30 years ago, old fashioned menu board, booths out of the 70s, and a large selection of food. Their speciality is the Italian Hoagie, and it is voted the best in the area year after year. I usually order the burger, while the patties are obviously cooked from frozen, all of the other ingredients are very fresh. Overall, its a good alternative to Subway, which is down the road."

def text2words(text):
    import re
    def clean_text(text):
        return re.sub(r'[.;:,!\'"]', ' ', unicode(text).lower())
    return filter(lambda x: x!='', clean_text(text).split(' '))

text2words(text)

def json_review(s):
    import json
    r = json.loads(s.strip())
    return (r['stars'], r['text'])

rtrain_rdd.map(json_review).take(10)

##word_train_rdd = rtrain_rdd.flatMap(lambda r: [(r[0], w) for w in text2words(r[1])])
word_train_rdd = rtrain_rdd.map(json_review).flatMap(lambda r: [(r[0], w) for w in text2words(r[1])])

word_train_rdd.take(10) ## .groupByKey().take(10)

import numpy as np

def stars_one_hot(r):
    import numpy as np
    s = np.zeros(5)
    s[r[0]-1] = 1
    return (r[1], s)

words_train_oh_rdd = word_train_rdd.map(stars_one_hot).take(10)

def sum_one_hot_stars(vs):
    import numpy as np
    n = 0
    sum_s = np.zeros(5)
    for v in vs:
        n += 1
        sum_s += v
    return (sum_s, n)

word_count = word_train_rdd    .map(stars_one_hot).groupByKey()    .map(lambda (k,vs): (k, sum_one_hot_stars(vs)))

word_count.take(4)

rtrain_rdd.map(json_review).map(lambda t: (t[0], 1.0)).reduceByKey(lambda a,b: a+b).take(10)

def calc_perc_freq(t):
    import numpy as np
    freq = t[1][0]
    tot = t[1][1]
    freq/float(tot)
    return (t[0], freq/float(tot))



word_count.sortBy(lambda r: r[1][1], ascending=False).take(10)

word_count.sortBy(lambda r: r[1][1], ascending=False).map(calc_perc_freq).take(10)

def entropy(t):
    import numpy as np
    p = t[1]
    return (t[0], t[1], -np.sum(np.log(p)*p))

word_count.sortBy(lambda r: r[1][1], ascending=False).map(calc_perc_freq).map(entropy).take(10)

word_freq = word_count.map(calc_perc_freq).map(entropy)

word_freq.filter(lambda x: ~np.isnan(x[2])).sortBy(lambda x: x[2], ascending=True).take(10)

word_freq.filter(lambda x: ~np.isnan(x[2])).sortBy(lambda x: x[2], ascending=False).take(10)

import numpy as np
x = np.array([ 0.10591722,  0.08672078,  0.13137777,  0.27420604,  0.4017782 ])

-np.sum(np.log(x)*x)

np.array([1,0])*np.array([3,7])



t = np.zeros(5)
t[2] = 1







s+t

np.sum([s, t])







unicode.lower

