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

# Register the DataFrame as a table
df.registerTempTable("mytable")

# Peform a simple select from the table
results = sqlContext.sql("select * from mytable")

# Convert the results to a Pandas DataFrame for easy viewing
results.toPandas()

# Perform a query with a where clause and order by
results = sqlContext.sql("select * from mytable where numbers < 8 order by numbers desc")

# Convert the results to a Pandas DataFrame for easy viewing
results.toPandas()

# Perform a more complex query on the table
results = sqlContext.sql("select letters, count(*) as count, avg(numbers) as avg, sum(numbers) as sum from mytable group by letters")

# Convert the results to a Pandas DataFrame for easy viewing
results.toPandas()

# Stop the context when you are done with it. When you stop the SparkContext resources 
# are released and no further operations can be performed within that context
sc.stop()

