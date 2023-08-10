import pyspark
spark = pyspark.SparkContext("local[*]")

numberRDD = spark.parallelize(range(1, 10000))

# filter numberRDD, keeping only even numbers
evens = numberRDD.filter(lambda x: x % 2 == 0)

# produce an RDD by doubling every element in numberRDD
doubled = numberRDD.map(lambda x: x * 2)

# filter numberRDD, keeping only multiples of five
fives = numberRDD.filter(lambda x: x % 5 == 0)

# return an RDD of the elements in both evens and fives
tens = evens.intersection(fives)
sortedTens = tens.sortBy(lambda x: x)

(evens.count(), doubled.count())

# note that we may not get results in order!
tens.take(5)

# ...unless we sort
sortedTens.take(5)

# we can take a sample from an RDD (with or without replacement)
sortedTens.takeSample(False, 10)

sortedTens.reduce(lambda x, y: max(x, y))

from pyspark.sql import SQLContext
sqlc = SQLContext(spark)

df = sqlc.read.parquet("/msgs.parquet")
df.printSchema()

df.groupBy('category').count().orderBy('count', ascending=False).show()

df.count()

df.show(10)

msgRDD = df.select("msg").rdd.map(lambda x: x[0])
# structs = sqlc.jsonRDD(msgRDD)
# structs.printSchema()

import json

# define the fields we want to keep
interesting_fields = ['agent', 'author', 'copr', 'user', 'msg', 'meeting_topic', 'name', 'owner', 'package']

# describe the return type of our user-defined function
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
resultType = StructType([StructField(f, StringType(), nullable=True) for f in interesting_fields])

# this is the body of our first user-defined function, to restrict us to a subset of fields
def trimFieldImpl(js):
    try:
        d = json.loads(js)
        return [d.get(f) for f in interesting_fields]
    except:
        # return an empty struct if we fail to parse this message
        return [None] * len(interesting_fields)
    
from pyspark.sql.functions import udf

# register trimFieldImpl as a user-defined function
trimFields = udf(trimFieldImpl, resultType)

trimmedDF = df.withColumn("msg", trimFields("msg"))

trimmedDF.printSchema()

def getComments(js):
    try:
        d = json.loads(js)
        cs = d.get('comment', []) + d.get('update', {}).get('comments', [])
        notes = 'notes' in d and [d['notes']] or []
        return [c['text'] for c in cs if c.has_key('text')] + []
    except:
        return []

commentsRDD = msgRDD.flatMap(lambda js: getComments(js))

# turn comments into sequences of words.  don't bother stripping punctuation or stemming #yolo
wordSeqs = commentsRDD.map(lambda s: s.split())

# actually train a model

from pyspark.mllib.feature import Word2Vec

w2v = Word2Vec()
model = w2v.fit(wordSeqs)

# find synonyms for a given word
synonyms = model.findSynonyms('works', 5)

for word, distance in synonyms:
    print("{}: {}".format(word, distance))

# see what words are in the model

model.getVectors().keys()



