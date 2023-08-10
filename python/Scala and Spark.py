import pyspark
sc = pyspark.SparkContext('local[*]')
print('Spark context:',sc)

sc.parallelize([1,2,3])

sc.parallelize([1,2,3]).collect()

an_rdd = sc.parallelize([1,2,3])
an_rdd.collect()

rdd_line = sc.textFile("Dance_5lines.txt")

rdd_line.take(2)

rdd_line.collect()

rdd_line.count()

rdd_line.takeSample(withReplacement=False, num=2)

rdd_line.map(lambda line:  len(line)).take(5)

rdd_line.map(lambda line:  len(line)).reduce(lambda x, y: x+y)

rdd_line.filter(lambda line: "2015" in line).collect()

"February 26,         2016".split()

rdd_line.map(lambda line: line.split()).take(3)

rdd_line.flatMap(lambda line: line.split()).take(7)

rdd_word = rdd_line.flatMap(lambda line: line.split())

rdd_word.take(7)

rdd_word_1 = rdd_word.map(lambda word: (word,1))
rdd_word_1.take(5)

rdd_word_count = rdd_word_1.reduceByKey(lambda count_1, count_2: count_1 + count_2)

rdd_word_count.collect()

rdd_word_count.filter(lambda key_val: key_val[1]>1).collect()

from pyspark.sql import SQLContext, Row
sqlContext = SQLContext(sc)
sqlContext

iris_text = sc.textFile("iris_noheader.csv")
iris_text.take(3)

iris_line = iris_text.map(lambda l: l.split(","))
iris_line.take(3)

iris_row  = iris_line.map(lambda p: Row(SepalLength=float(p[0]),
                                        SepalWidth=float(p[1]),
                                        PetalLength=float(p[2]),
                                        PetalWidth=float(p[3]),
                                        Species=p[4]))
print('iris_row:',iris_row)
iris_row.take(3)

from pyspark.mllib.linalg.distributed import RowMatrix
rows = iris_line.map(lambda line: (line[0], line[1], line[2], line[3]))
mat = RowMatrix(rows)
mat.numRows(), mat.numCols()

from pyspark.mllib.stat import Statistics

# Compute column summary statistics.
summary = Statistics.colStats(mat)
print(summary.mean())
print(summary.variance())
print(summary.numNonzeros())

iris_df = sqlContext.createDataFrame(iris_row)
iris_df.printSchema()
iris_df.take(3)

iris_df.select("PetalLength","PetalWidth","Species").filter(iris_df['PetalLength'] > 1.5).show()

iris_df.describe("PetalLength","PetalWidth").show()

iris_df.groupBy("Species").count().show()

from pyspark.sql import functions as pf
iris_df.select("PetalLength","Species"). groupBy("Species"). agg(pf.mean(iris_df.PetalLength)). collect()

iris_df.registerTempTable("iris")

sqlContext.sql("SELECT * from iris")

sqlContext.sql("SELECT * from iris").show()

sqlContext.sql("SELECT * from iris where SepalWidth > 4.0").show()

from pyspark.mllib.stat import Statistics

# Compute column summary statistics.
summary = Statistics.colStats(iris_df)
print(summary.mean())
print(summary.variance())
print(summary.numNonzeros())



