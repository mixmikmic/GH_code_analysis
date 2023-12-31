import pandas as pd
import numpy as np
import glob
import os
import bz2
import codecs
import json
import matplotlib.pylab as plt
import uuid
import random
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, DoubleType, StringType
from pyspark.sql.functions import to_utc_timestamp, unix_timestamp, from_unixtime
get_ipython().run_line_magic('matplotlib', 'inline')

conf = SparkConf()

conf.setMaster('local[8]')

sc = SparkContext.getOrCreate(conf=conf)
sql_sc = SQLContext(sc)

schema = StructType([
        StructField('VendorID',IntegerType(),True),
        StructField('tpep_pickup_datetime',TimestampType(),True),
        StructField('tpep_dropoff_datetime',TimestampType(),True),
        StructField('passenger_count',IntegerType(),True),
        StructField('trip_distance',DoubleType(),True),
        StructField('pickup_longitude',DoubleType(),True),
        StructField('pickup_latitude',DoubleType(),True),
        StructField('RateCodeID',IntegerType(),True),
        StructField('store_and_fwd_flag',StringType(),True),
        StructField('dropoff_longitude',DoubleType(),True),
        StructField('dropoff_latitude',DoubleType(),True),
        StructField('payment_type',IntegerType(),True),
        StructField('fare_amount',DoubleType(),True),
        StructField('extra',DoubleType(),True),
        StructField('mta_tax',DoubleType(),True),
        StructField('tip_amount',DoubleType(),True),
        StructField('tolls_amount',DoubleType(),True),
        StructField('improvement_surcharge',DoubleType(),True),
        StructField('total_amount',DoubleType(),True),
    ])

infilename = 'yellow_tripdata_2015-03.csv'
#infilename = 'sample.csv'
raw_sdf = (sql_sc
    .read
    .csv(infilename, header=True, schema=schema, timestampFormat='yyyy-MM-dd HH:mm:ss'))

clean_sdf = raw_sdf.withColumn('tpep_pickup_timestamp_ms',  unix_timestamp(raw_sdf['tpep_pickup_datetime' ])*1000 + 5*60*60*1000)
clean_sdf = clean_sdf.withColumn('tpep_dropoff_timestamp_ms', unix_timestamp(raw_sdf['tpep_dropoff_datetime'])*1000 + 5*60*60*1000)

clean_sdf.limit(1).toPandas().info()

clean_sdf.limit(5).toPandas()

end_timestamp = pd.Timestamp('2015-03-03 00:00:00').tz_localize('Etc/GMT+5')
filtered_sdf = clean_sdf.filter('tpep_dropoff_timestamp_ms <= %d' % int(end_timestamp.value / 1e6) )

filtered_sdf.count()

filtered_sdf.limit(5).toPandas()

def create_events(trip_record):
    events = []
    trip_id = str(uuid.uuid4())
    pickup_datetime = trip_record.tpep_pickup_timestamp_ms
    dropoff_datetime = trip_record.tpep_dropoff_timestamp_ms
    events.append({
        'event_type': 'pickup',
        'trip_id': trip_id,
        'trip_fraction': 0.0,
        'trip_duration_minutes': 0.0,
        'VendorID': trip_record.VendorID,
        'timestamp': pickup_datetime,
        'passenger_count': trip_record.passenger_count,
        'trip_distance': 0.0,
        'lat': trip_record.pickup_latitude,
        'lon': trip_record.pickup_longitude,
        'RateCodeID': trip_record.RateCodeID,
        'store_and_fwd_flag': trip_record.store_and_fwd_flag,
        'payment_type': trip_record.payment_type,
        'fare_amount': 0.0,
        'extra': 0.0,
        'mta_tax': 0.0,
        'tip_amount': 0.0,
        'tolls_amount': 0.0,
        'improvement_surcharge': 0.0,
        'total_amount': 0.0,
    })
    
    # Create route events every 1 minute.
    # Assume that route is a straight line from pickup to dropoff.
    # Assume that dollar amounts are spread uniformly throughout the trip.
    trip_duration = dropoff_datetime - pickup_datetime
    report_period_mean = 1*60*1000
    report_timestamp = pickup_datetime
    while True:
        # The next report period is a random number. This allows us to get timestamps with non-zero milliseconds.
        report_period = random.randint(0, 2*report_period_mean)
        report_timestamp += report_period
        if report_timestamp >= dropoff_datetime:
            break
        trip_fraction = (report_timestamp - pickup_datetime) / trip_duration
        events.append({
            'event_type': 'route',
            'trip_id': trip_id,
            'trip_fraction': trip_fraction,
            'trip_duration_minutes': trip_duration * trip_fraction / (60*1000),
            'VendorID': trip_record.VendorID,
            'timestamp': report_timestamp,
            'passenger_count': trip_record.passenger_count,
            'trip_distance': trip_record.trip_distance * trip_fraction,            
            'lat': trip_record.pickup_latitude + (trip_record.dropoff_latitude - trip_record.pickup_latitude) * trip_fraction,
            'lon': trip_record.pickup_longitude + (trip_record.dropoff_longitude - trip_record.pickup_longitude) * trip_fraction,
            'RateCodeID': trip_record.RateCodeID,
            'store_and_fwd_flag': trip_record.store_and_fwd_flag,
            'payment_type': trip_record.payment_type,            
            'fare_amount': trip_record.fare_amount * trip_fraction,
            'extra': trip_record.extra * trip_fraction,
            'mta_tax': trip_record.mta_tax * trip_fraction,
            'tip_amount': trip_record.tip_amount * trip_fraction,
            'tolls_amount': trip_record.tolls_amount * trip_fraction,
            'improvement_surcharge': trip_record.improvement_surcharge * trip_fraction,
            'total_amount': trip_record.total_amount * trip_fraction,
        })

    events.append({
        'event_type': 'dropoff',
        'trip_id': trip_id,
        'trip_fraction': 1.0,
        'trip_duration_minutes': trip_duration / (60*1000),
        'VendorID': trip_record.VendorID,
        'timestamp': dropoff_datetime,
        'passenger_count': trip_record.passenger_count,
        'trip_distance': trip_record.trip_distance,
        'lat': trip_record.dropoff_latitude,
        'lon': trip_record.dropoff_longitude,
        'RateCodeID': trip_record.RateCodeID,
        'store_and_fwd_flag': trip_record.store_and_fwd_flag,
        'payment_type': trip_record.payment_type,
        'fare_amount': trip_record.fare_amount,
        'extra': trip_record.extra,
        'mta_tax': trip_record.mta_tax,
        'tip_amount': trip_record.tip_amount,
        'tolls_amount': trip_record.tolls_amount,
        'improvement_surcharge': trip_record.improvement_surcharge,
        'total_amount': trip_record.total_amount,
    })
    rows = [Row(**event) for event in events]
    return rows

get_ipython().run_cell_magic('time', '', 'all_events_rdd = filtered_sdf.rdd.flatMap(create_events)')

all_events_rdd.take(2)

get_ipython().run_cell_magic('time', '', 'all_events_sdf = sql_sc.createDataFrame(all_events_rdd)')

all_events_sdf.show()

all_events2_sdf = all_events_sdf.withColumn('timestamp_str',  from_unixtime(all_events_sdf['timestamp' ] / 1000))

all_events2_sdf.show()

get_ipython().run_cell_magic('time', '', "sorted_sdf = all_events2_sdf.orderBy('timestamp')")

get_ipython().run_cell_magic('time', '', "sorted_sdf.write.mode('overwrite').format('json').save('data.json')")



