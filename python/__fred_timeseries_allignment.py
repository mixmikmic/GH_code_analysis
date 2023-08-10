from datetime import datetime

from pyspark import SparkContext, SQLContext
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType, IntegerType

from sparkts.datetimeindex import uniform, BusinessDayFrequency
from sparkts.timeseriesrdd import time_series_rdd_from_observations

import sparkts.datetimeindex as dt

dt.DayFrequency

def lineToRow(line):
    (year, month, day, symbol, volume, price) = line.split("\t")
    # Python 2.x compatible timestamp generation
    dt = datetime(int(year), int(month), int(day))
    return (dt, symbol, float(price))

def loadObservations(sparkContext, sqlContext, path):
    textFile = sparkContext.textFile(path)
    rowRdd = textFile.map(lineToRow)
    schema = StructType([
        StructField('timestamp', TimestampType(), nullable=True),
        StructField('symbol', StringType(), nullable=True),
        StructField('price', DoubleType(), nullable=True),
    ])
    return sqlContext.createDataFrame(rowRdd, schema);

get_ipython().system('wget https://raw.githubusercontent.com/sryza/spark-ts-examples/master/data/ticker.tsv')

tickerObs = loadObservations(sc, sqlContext, "/Users/guillermobreto/Downloads/spark-timeseries/DOCS_REPO/spark-timeseries/ticker.tsv")

tickerObs.select("timestamp").take(3)

tickerObs.show(3, truncate=False)

tickerObs.printSchema()


# Create an daily DateTimeIndex over August and September 2015
freq = BusinessDayFrequency(1, 1, sc)
dtIndex = uniform(start='2015-08-03T00:00-04:00', end='2015-09-22T00:00-04:00', freq=freq, sc=sc)


tickerTsrdd = time_series_rdd_from_observations(dtIndex, tickerObs, "timestamp", "symbol", "price")

tickerTsrdd.take(2)


# Count the number of series (number of symbols)
print(tickerTsrdd.count())

# Impute missing values using linear interpolation
filled = tickerTsrdd.fill("linear")

# Compute return rates
returnRates = filled.return_rates()

filled.take(2)

# Durbin-Watson test for serial correlation, ported from TimeSeriesStatisticalTests.scala
def dwtest(residuals):
    residsSum = residuals[0] * residuals[0]
    diffsSum = 0.0
    i = 1
    while i < len(residuals):
        residsSum += residuals[i] * residuals[i]
        diff = residuals[i] - residuals[i - 1]
        diffsSum += diff * diff
        i += 1
    return diffsSum / residsSum

# Compute Durbin-Watson stats for each series
# Swap ticker symbol and stats so min and max compare the statistic value, not the
# ticker names.
dwStats = returnRates.map_series(lambda row: (row[0], [dwtest(row[1])])).map(lambda x: (x[1], x[0]))

print(dwStats.min())
print(dwStats.max())

rdd = sc.wholeTextFiles("/Users/guillermobreto/Downloads/fred_timeseries_project/data/fred_codes/")
print(rdd.count())
from pyspark.sql.functions import explode
import pyspark.sql.functions as f
from pyspark.sql.functions import udf
from datetime import datetime

from pyspark import SparkContext, SQLContext
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType

from sparkts.datetimeindex import uniform, BusinessDayFrequency
from sparkts.timeseriesrdd import time_series_rdd_from_observations

import numpy as np
import pandas as pd

from sparkts.datetimeindex import DayFrequency

freq = DayFrequency(1,sc)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#freq = BusinessDayFrequency(1, 1, sc)
dtIndex = uniform(start='2005-02-01T00:00-05:00', end='2005-06-01T00:00-05:00', freq=freq, sc=sc)

rdd_df = rdd.map(lambda r: (r[0].split("/")[-1].strip(".csv"),filter(None, r[1].split("\n")[1:]))).toDF(["symbol","v"])

rdd_df.select("symbol").distinct().count()

rdd_df=rdd_df.limit(100)
rdd_df.show(3)

rdd_df.select("symbol").distinct().count()

rdd_df_exp =  rdd_df.select([rdd_df.symbol,explode(rdd_df.v).alias("DATA-VALUE")])

valueUdf = udf(lambda s: float(s.split(",")[1]), DoubleType())
dateUdf = udf(lambda s: s.split(",")[0], StringType())
new_df =rdd_df_exp.withColumn("Date", (f.to_date(f.lit(dateUdf(rdd_df_exp["DATA-VALUE"]))).cast(TimestampType())))
new_df =new_df.withColumn("price", valueUdf(new_df["DATA-VALUE"]))

new_df.show(3)

new_df.select("symbol").distinct().count()

freq = DayFrequency(1,sc)
dtIndex = uniform(start='2015-01-01T00:00-05:00', end='2016-10-01T00:00-05:00', freq=freq, sc=sc)

dates = ("2015-01-01",  "2016-10-01")
date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
df_filtered = new_df.where((new_df.Date > date_from) & (new_df.Date < date_to))

df_filtered.show(3)

df = df_filtered.select(["symbol", "Date", "price"])
df = df.withColumnRenamed("Date", "timestamp")

df.show(2, truncate=False)

tickerTsrdd = time_series_rdd_from_observations(dtIndex, df, "timestamp", "symbol", "price")

tickerTsrdd.take(3)

filled = tickerTsrdd.fill("linear")

filled.take(2)

previous = filled.fill("previous")

previous.take(3)

nearest = previous.fill("nearest")

nearest.take(1)

rr = nearest.return_rates()

rr = rr.fill("linear")

rr = rr.map(lambda ts: (ts[0], np.nan_to_num(ts[1])))

rr.take(1)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def shifting(a, delta):
    from scipy.ndimage.interpolation import shift
    return shift(a, delta, cval=np.NaN)

ma = rr.map(lambda row:  (row[0], moving_average(row[1], 10)))

ma.take(2)

sh = rr.map(lambda row:  (row[0] + '_shift', np.nan_to_num(shifting(row[1], 1))))
ma = rr.map(lambda row:  (row[0] + "_mov_avg", moving_average(row[1])))

sh.take(1)

total = sc.union([rr, ma, sh])

lenghts = total.map(lambda ts: len(ts[1]))

colLength = np.array(lenghts.collect())

colLength.min()

total.count()

from pyspark.mllib.linalg import Vectors
total_df = total.map(lambda x: Row(symbol=x[0], feat=Vectors.dense(x[1]))).map(lambda x: [x[1], x[0]]).toDF(["symbol","feat"])

total_df.show(3)

udfToArray = udf(lambda s: len(s), IntegerType())

total_df.printSchema()

total_df= total_df.withColumn("length",udfToArray(total_df.feat) )

total_df.show(2)

total_df_clean = total_df.filter("length=639")

limited =  total_df_clean.map(lambda ts: [ts[0], filter(None,[float(l) for l in ts[1].toArray()])])

ts = limited.toDF(["Symbol", "ts"])

ts.show(1)

ts_exploded = ts.select([ts.Symbol,explode(ts.ts).alias("values")])

ts_exploded.show(10)

from pyspark.sql.functions import monotonicallyIncreasingId

# This will return a new DF with all the columns + id
res = ts_exploded.withColumn("index", monotonicallyIncreasingId())

res.show(10)

pivoted = res.groupBy("index").pivot("Symbol").sum("values")

from pyspark.sql.window import Window
ranked = res.select("Symbol", "index", "values",
     f.rowNumber()
     .over(Window
           .partitionBy("Symbol")
           .orderBy(f.col("index").desc())
            )
     .alias("rank")
    )

pivoted.select(["index","FRED_00XALCCHM086NEST","FRED_00XALCCHM086NEST_shift"]).show()

import datetime
import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)

start = datetime.datetime(2008, 1, 1)
end = datetime.datetime(2014, 8, 15)
#trim_start = '2000-01-01'
#trim_end = '2014-08-15'

sp =  pd.io.data.get_data_yahoo('^GSPC', start, end)
#sp.head(10)

sp.columns.values[-1] = 'AdjClose'
sp.columns = sp.columns + '_SP500'
sp['Return_SP500'] = sp['AdjClose_SP500'].pct_change()
sp.columns



nasdaq =  pd.io.data.get_data_yahoo('^IXIC', start, end)
#nasdaq.head()

nasdaq.columns.values[-1] = 'AdjClose'
nasdaq.columns = nasdaq.columns + '_Nasdaq'
nasdaq['Return_Nasdaq'] = nasdaq['AdjClose_Nasdaq'].pct_change()
nasdaq.columns

treasury =  pd.io.data.get_data_yahoo('^FVX', start, end)

treasury.columns.values[-1] = 'AdjClose'
treasury.columns = treasury.columns + '_Treasury'
treasury['Return_Treasury'] = treasury['AdjClose_Treasury'].pct_change()
treasury.columns

datasets = [sp, nasdaq, treasury]


to_be_merged = [nasdaq[['Return_Nasdaq']],
                treasury[['Return_Treasury']],
]
                
finance = sp[['Return_SP500']].join(to_be_merged, how = 'outer')

finance.head()



