# required spark imports
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

conf = SparkConf().setMaster("local").set("spark.driver.memory", "4g")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)

ratings = spark.read.option("header","true").csv("../data/ratings*.csv.gz")

ratings.show(5)

movies = spark.read.option("header","true").csv("../data/movies.csv.gz")

movies.show(5, False)

tags = spark.read.option("header","true").csv("../data/tags.csv.gz")

tags.show(5)

ratings.createOrReplaceTempView("ratings")
movies.createOrReplaceTempView("movies")
tags.createOrReplaceTempView("tags")

from pyspark.sql.functions import regexp_extract
movies_year = movies.withColumn("Year",regexp_extract("title",'^(.*) \\(([0-9 \\-]*)\\)$',2))
movies_year.show(5,False)
movies_year.createOrReplaceTempView("movies_year")

m_yr = spark.sql("select year, count(1) as count from movies_year group by year order by year").toPandas()

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import pandas as pd

m_yr.plot(x='year',y='count',kind='line')

spark.sql("select title, count(*) from movies m, ratings r where m.movieId = r.movieId group by title order by 2 desc").show(5)

spark.sql("select title, count(*) from movies m, ratings r where m.movieId = r.movieId group by title order by 2 desc").explain(True)

spark.sql("select title, avg(rating) as avg_rating from movies m, ratings r             where m.movieId = r.movieId             group by title             order by 2 desc").show(5, False)

spark.sql("select title, avg(rating) as avg_rating, count(*) as count from movies m, ratings r             where m.movieId = r.movieId             group by title             order by 2 desc").show(5, False)

spark.sql("select title, avg(rating) as avg_rating, count(*) as count from movies m, ratings r             where m.movieId = r.movieId             group by title             having count(*) > 100             order by 2 desc").show(20, False)

avg_ratings = spark.sql("select year, title, avg(rating) as avg_rating, count(*) as count from movies_year m, ratings r where m.movieId = r.movieId group by year, title having count(*) > 100")
avg_ratings.createOrReplaceTempView("avg_ratings")
spark.sql("select a.year, a.title, avg_rating from avg_ratings a,             (select year, max(avg_rating) as max_rating from avg_ratings group by year) m             where a.year = m.year             and a.avg_rating = m.max_rating             and a.year > 2000             order by year").show(20, False)

get_ipython().system('pip install --user wordcloud')

from wordcloud import WordCloud, STOPWORDS

children_tags = spark.sql("select tag from tags t, movies m where t.movieId = m.movieId and genres like '%Children%'").toPandas()

# Generate a word cloud image
wordcloud = WordCloud(width=1200,height=600).generate(' '.join(children_tags['tag']))

# Display the generated image
plt.figure(figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")

