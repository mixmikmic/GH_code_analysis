from pyspark.sql import SparkSession
from datetime import datetime
path = "/Users/josephgartner/Desktop/data/"

spk = SparkSession.builder.master("local").getOrCreate()
df = spk.read.json(path)

df.printSchema()

df.select('lang', 'user.name').show(5)

df.filter(df['geo'].isNotNull()).select('geo').take(1)[0].geo.coordinates[0]

n_tot = df.count()
print "There are {} tweets in this sample.".format(n_tot)

n_w_geo = df.filter(df['geo'].isNotNull()).count()
print "{0:.6f}% of which have geo tags".format(float(n_w_geo)/n_tot)

n_non_eng = df.filter(df['lang']!='en').filter(df['lang']!='und').count()
print "{0:.4f}% of which aren't english".format(float(n_non_eng)/n_tot)

df_mod = df.filter(df['geo'].isNotNull()).withColumn('geo-first', df.geo.coordinates[0] + 1)
df_mod.select('geo.coordinates', 'geo-first').show(5)

#Verify transformation format is OK
print datetime.strptime(df.select('created_at').take(1)[0].created_at, "%a %b %d %H:%M:%S +0000 %Y")

try:
    df.withColumn("created_at_dt", datetime.strptime(df["created_at"], "%a %b %d %H:%M:%S +0000 %Y"))
except TypeError as te:
    print te

from pyspark.sql.functions import udf
from pyspark.sql.types import TimestampType, BooleanType

make_dt = udf(lambda date_string: datetime.strptime(date_string, "%a %b %d %H:%M:%S +0000 %Y"), TimestampType())
df_mod = df.withColumn("dt_created_at", make_dt(df['created_at']))
df_mod.select("created_at", "dt_created_at").show(5)

def target_market(dt, lng):
    if dt.minute == 23 and lng != 'en':
        return True
    else:
        return False

is_target = udf(lambda dt, lng: target_market(dt, lng), BooleanType())
df_mod_filtered = df_mod.withColumn('is_target', is_target(df_mod['dt_created_at'], df_mod['lang']))
df_mod_filtered.select('is_target', 'dt_created_at', 'lang').show(5)

df_mod_filtered.filter(df_mod_filtered['is_target']==True).select('is_target', 'dt_created_at', 'lang').show(5)

