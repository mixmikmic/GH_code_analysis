get_ipython().system('hadoop fs -ls /user/cloudera/movielens')

movies = spark.read.format("csv").option("header", True).load("/user/cloudera/movielens/movies.csv")
movies.show()

type(movies)

movies.printSchema()

movies = spark.read.format("csv").options(header = True, inferSchema = True).load("/user/cloudera/movielens/movies.csv")

movies.printSchema()

ratings = spark.read.format("csv").options(header = True, inferSchema = True).load("/user/cloudera/movielens/ratings.csv")

ratings.printSchema()

ratings.show()

from pyspark.sql.functions import *

(
    ratings
    .groupBy("movieId")
    .agg(avg("rating").alias("avg_rating"), count("rating").alias("rating_count"))
    .alias("t1")
    .join(movies.alias("t2"), col("t1.movieId") == col("t2.movieId"))
    .filter("rating_count > 100")
    .orderBy(desc("avg_rating"))
).show()

movies.createOrReplaceTempView("movies")

sql("show tables").show()

ratings.createOrReplaceTempView("ratings")

sql("show tables").show()

df = sql("""
select t1.movieId, t1.title, avg(t2.rating) avg_rating, count(1) rating_count
from movies t1 join ratings t2 on t1.movieId = t2.movieId 
group by t1.movieId, t1.title 
having rating_count > 100
order by avg_rating desc
""")
df.show()

df.write.format("json").save("ml-agg")

df.rdd.getNumPartitions()

spark.sparkContext.getConf().getAll()

spark.conf.set("spark.sql.shuffle.partitions", "5")

df = sql("""
select t1.movieId, t1.title, avg(t2.rating) avg_rating, count(1) rating_count
from movies t1 join ratings t2 on t1.movieId = t2.movieId 
group by t1.movieId, t1.title 
having rating_count > 100
order by avg_rating desc
""")
df.rdd.getNumPartitions()

df.coalesce(1).rdd.getNumPartitions()

df.coalesce(1).write.mode("overwrite").format("json").save("ml-agg")

ml_agg = spark.read.format("json").load("ml-agg")
ml_agg.show()

ml_agg = spark.read.json("ml-agg")
ml_agg.show()

df.coalesce(1).write.mode("overwrite").save("ml-agg-parquet")

spark.read.load("ml-agg-parquet").show()

sql("select * from parquet.`/user/cloudera/ml-agg-parquet`").show()

sql("select * from parquet.`/user/cloudera/ml-agg-parquet`").explain()

df.explain()

df.write.saveAsTable("ml_agg")

df = (
    spark.read.format("jdbc")
    .option("url", "jdbc:mysql://localhost:3306/retail_db")
    .option("driver", "com.mysql.jdbc.Driver")
    .option("dbtable", "orders")
    .option("user", "root")
    .option("password", "cloudera")
    .load())
df.show()

df.count()

df.write.saveAsTable("orders")

sql("show tables").show()

(
spark
.read
.format("jdbc")
.option("url", "jdbc:mysql://localhost:3306/retail_db")
.option("driver", "com.mysql.jdbc.Driver")
.option("dbtable", "customers")
.option("user", "root")
.option("password", "cloudera")
.load()
.createOrReplaceTempView("customers")
)

sql("show tables").show()

sql("""
select t1.customer_id, count(*) complete_count  
from customers t1 join orders t2 on t1.customer_id = t2.order_customer_id 
where t2.order_status = 'COMPLETE' 
group by t1.customer_id order by complete_count desc limit 10
""").show()

sql("show tables").show()

sql("select * from movies").show()

spark.udf.register("myupper", lambda s: s.upper())

sql("select *, myupper(title) from movies").show()

movies.select("title", expr("myupper(title)").alias("title_upper")).show()

movies.selectExpr("title", "myupper(title)").toDF("title", "title_upper").show()



