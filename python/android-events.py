import datetime as dt
import pandas as pd
import ujson as json
from pyspark.sql.types import *

from moztelemetry import get_pings, get_pings_properties, get_one_ping_per_client

get_ipython().magic('pylab inline')

def dedupe_pings(rdd):
    return rdd.filter(lambda p: p["meta/clientId"] is not None)              .map(lambda p: (p["meta/documentId"], p))              .reduceByKey(lambda x, y: x)              .map(lambda x: x[1])

def safe_str(obj):
    """ return the byte string representation of obj """
    if obj is None:
        return unicode("")
    return unicode(obj)

def transform(ping):    
    output = []

    # These should not be None since we filter those out & ingestion process adds the data
    clientId = ping["meta/clientId"]
    submissionDate = dt.datetime.strptime(ping["meta/submissionDate"], "%Y%m%d")

    events = ping["payload/UIMeasurements"]
    if events and isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and "type" in event and event["type"] == "event":
                if "timestamp" not in event or "action" not in event or "method" not in event or "sessions" not in event:
                    continue

                # Verify timestamp is a long, otherwise ignore the event
                timestamp = None
                try:
                    timestamp = long(event["timestamp"])
                except:
                    continue

                # Force all fields to strings
                action = safe_str(event["action"])
                method = safe_str(event["method"])

                # The extras is an optional field
                extras = unicode("")
                if "extras" in event and safe_str(event["extras"]) is not None:
                    extras = safe_str(event["extras"])

                sessions = {}
                experiments = []
                for session in event["sessions"]:
                    if "experiment.1:" in session:
                        experiments.append(safe_str(session[13:]))
                    elif "firstrun.1" in session:
                        sessions[unicode("firstrun")] = 1
                    elif "awesomescreen.1" in session:
                        sessions[unicode("awesomescreen")] = 1
                    elif "reader.1" in session:
                        sessions[unicode("reader")] = 1

                output.append([clientId, submissionDate, timestamp, action, method, extras, json.dumps(sessions.keys()), json.dumps(experiments)])

    return output

def dedupe_events(rdd):
    return rdd.map(lambda p: (p[0] + safe_str(p[2]) + p[3] + p[4], p))              .reduceByKey(lambda x, y: x)              .map(lambda x: x[1])

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
                                              "payload/UIMeasurements"])

        subset = dedupe_pings(subset)
        print subset.first()

        rawEvents = subset.flatMap(transform)
        print "\nRaw count: " + str(rawEvents.count())
        print rawEvents.first()

        uniqueEvents = dedupe_events(rawEvents)
        print "\nUnique count: " + str(uniqueEvents.count())
        print uniqueEvents.first()

        s3_output = "s3n://net-mozaws-prod-us-west-2-pipeline-analysis/mobile/android_events"
        s3_output += "/v1/channel=" + channel + "/submission=" + day.strftime("%Y%m%d") 
        schema = StructType([
            StructField("clientid", StringType(), False),
            StructField("submissiondate", TimestampType(), False),
            StructField("ts", LongType(), True),
            StructField("action", StringType(), True),
            StructField("method", StringType(), True),
            StructField("extras", StringType(), True),
            StructField("sessions", StringType(), True),
            StructField("experiments", StringType(), True)
        ])
        grouped = sqlContext.createDataFrame(uniqueEvents, schema)
        grouped.coalesce(1).write.parquet(s3_output)

    day += dt.timedelta(1)

