from pyspark.sql import SparkSession
from datetime import datetime
path = "/Users/josephgartner/Desktop/data/"

spk = SparkSession.builder.master("local").getOrCreate()
df = spk.read.json(path)

df.createTempView("tweets")

#Selecting specific fields
df2 = spk.sql("SELECT lang, user.name FROM tweets")
df2.show(5)

#Remove null fields
df2 = spk.sql("SELECT lang, user.name, geo FROM tweets WHERE geo != NULL")
df2.show(5)

#Find specific field values
df2 = spk.sql("SELECT lang, user.name FROM tweets WHERE lang = 'en'")
df2.show(5)

#Perform basic transformations
df2 = spk.sql("SELECT geo.coordinates[0], geo.coordinates[0] + 1 FROM tweets WHERE geo != NULL")
df2.show(5)

#And basic groupings
df2 = spk.sql("SELECT lang, COUNT(*) FROM tweets GROUP BY lang")
df2.show(10)

def point_in_box(geo):
    if geo is None or geo.coordinates is None:
        return False
    if geo.coordinates[0] > 53 or geo.coordinates[0] < 52:
        return False
    if geo.coordinates[1] > -1 or geo.coordinates[1] < -2:
        return False
    return True

df2 = spk.sql("SELECT * FROM tweets WHERE geo != NULL")
temp = df2.take(5)

for tweet in temp:
    print point_in_box(tweet.geo)

df2 = spk.sql("SELECT * FROM tweets WHERE point_in_box(geo)")

from pyspark.sql.types import BooleanType

sqlContext.registerFunction("dist_pib", lambda geo: point_in_box(geo), returnType=BooleanType())

df2 = spk.sql("SELECT lang, user.name, geo FROM tweets WHERE dist_pib(geo)")
df2.show(5)



