import pyspark

# Create a SparkContext in local mode
sc = pyspark.SparkContext("local")

# Create a SqlContext from the SparkContext
sqlContext = pyspark.SQLContext(sc)

data = [
    (1, 'a'), 
    (2, 'b'), 
    (3, 'c'), 
    (4, 'd'), 
    (5, 'e'), 
    (6, 'a'), 
    (7, 'b'), 
    (8, 'c'), 
    (9, 'd'), 
    (10, 'e')
]

# Convert a local data set into a DataFrame
df = sqlContext.createDataFrame(data, ['numbers', 'letters'])

# Convert to a Pandas DataFrame for easy display
df.toPandas()

# Count the number of rows in the DataFrame
print df.count()

# View some rows
print df.take(3)

# Sort descending
descendingDf = df.orderBy(df.numbers.desc())

# View some rows
descendingDf.toPandas()

# Filter the DataFrame
filtered = df.where(df.numbers < 5)

# Convert to Pandas DataFrame for easy viewing
filtered.toPandas()

# Map the DataFrame into an RDD
rdd = df.map(lambda row: (row.numbers, row.numbers * 2))

# View some rows
print rdd.take(10)

# import some more functions
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import avg
from pyspark.sql.functions import sum

# Perform aggregations on the DataFrame
agg = df.agg(
    avg(df.numbers).alias("avg_numbers"), 
    sum(df.numbers).alias("sum_numbers"),
    countDistinct(df.numbers).alias("distinct_numbers"), 
    countDistinct(df.letters).alias('distinct_letters')
)

# Convert the results to Pandas DataFrame
agg.toPandas()

# View some summary statistics
df.describe().show()

# Stop the context when you are done with it. When you stop the SparkContext resources 
# are released and no further operations can be performed within that context
sc.stop()

