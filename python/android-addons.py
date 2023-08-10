import datetime as dt
import pandas as pd
import operator
import ujson as json
from pyspark.sql.types import *

from moztelemetry import get_pings, get_pings_properties, get_one_ping_per_client

get_ipython().magic('pylab inline')

def safe_str(obj):
    """ return the byte string representation of obj """
    if obj is None:
        return unicode("")
    return unicode(obj)

def dedupe_pings(rdd):
    return rdd.filter(lambda p: p["meta/clientId"] is not None)              .map(lambda p: (p["meta/documentId"], p))              .reduceByKey(lambda x, y: x)              .map(lambda x: x[1])

def dedupe_addons(rdd):
    return rdd.map(lambda p: (p[0] + safe_str(p[2]) + safe_str(p[3]), p))              .reduceByKey(lambda x, y: x)              .map(lambda x: x[1])

def clean(s):
    try:
        s = s.decode("ascii").strip()
        return s if len(s) > 0 else None
    except:
        return None

def transform(ping):    
    output = []

    # These should not be None since we filter those out & ingestion process adds the data
    clientId = ping["meta/clientId"]
    submissionDate = dt.datetime.strptime(ping["meta/submissionDate"], "%Y%m%d")

    addonset = {}
    addons = ping["environment/addons/activeAddons"]
    if addons is not None:
        for addon, desc in addons.iteritems():
            name = clean(desc.get("name", None))
            if name is not None:
                addonset[name] = 1

    persona = ping["environment/addons/persona"]

    if len(addonset) > 0 or persona is not None:
        addonarray = None
        if len(addonset) > 0:
            addonarray = json.dumps(addonset.keys())
        output.append([clientId, submissionDate, addonarray, persona])

    return output

channels = ["nightly", "aurora", "beta", "release"]

start = dt.datetime.now() - dt.timedelta(1)
end = dt.datetime.now() - dt.timedelta(1)

day = start
while day <= end:
    for channel in channels:
        print "\nchannel: " + channel + ", date: " + day.strftime("%Y%m%d")

        pings = get_pings(sc, app="Fennec", channel=channel,
                          submission_date=(day.strftime("%Y%m%d"), day.strftime("%Y%m%d")),
                          build_id=("20100101000000", "99999999999999"),
                          fraction=1)

        subset = get_pings_properties(pings, ["meta/clientId",
                                              "meta/documentId",
                                              "meta/submissionDate",
                                              "environment/addons/activeAddons",
                                              "environment/addons/persona"])

        subset = dedupe_pings(subset)
        print subset.first()

        rawAddons = subset.flatMap(transform)
        print "\nrawAddons count: " + str(rawAddons.count())
        print rawAddons.first()

        uniqueAddons = dedupe_addons(rawAddons)
        print "\nuniqueAddons count: " + str(uniqueAddons.count())
        print uniqueAddons.first()

        s3_output = "s3n://net-mozaws-prod-us-west-2-pipeline-analysis/mobile/android_addons"
        s3_output += "/v1/channel=" + channel + "/submission=" + day.strftime("%Y%m%d") 
        schema = StructType([
            StructField("clientid", StringType(), False),
            StructField("submissiondate", TimestampType(), False),
            StructField("addons", StringType(), True),
            StructField("lwt", StringType(), True)
        ])
        grouped = sqlContext.createDataFrame(uniqueAddons, schema)
        grouped.coalesce(1).write.parquet(s3_output)

    day += dt.timedelta(1)

