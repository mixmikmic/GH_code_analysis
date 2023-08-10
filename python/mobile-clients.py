import datetime as dt
import pandas as pd
import ujson as json
from pyspark.sql.types import *

from moztelemetry import get_pings, get_pings_properties

get_ipython().magic('pylab inline')

def dedupe_pings(rdd):
    return rdd.filter(lambda p: p["meta/clientId"] is not None)              .map(lambda p: (p["meta/documentId"], p))              .reduceByKey(lambda x, y: x)              .map(lambda x: x[1])

def transform(ping):
    # Should not be None since we filter those out.
    clientId = ping["meta/clientId"]

    # Added via the ingestion process so should not be None.
    submissionDate = dt.datetime.strptime(ping["meta/submissionDate"], "%Y%m%d")
    geoCountry = ping["meta/geoCountry"]

    profileDate = None
    profileDaynum = ping["profileDate"]
    if profileDaynum is not None:
        try:
            # Bad data could push profileDaynum > 32767 (size of a C int) and throw exception
            profileDate = dt.datetime(1970, 1, 1) + dt.timedelta(int(profileDaynum))
        except:
            profileDate = None

    # Create date should already be in ISO format
    creationDate = ping["creationDate"]
    if creationDate is not None:
        # This is only accurate because we know the creation date is always in 'Z' (zulu) time.
        creationDate = dt.datetime.strptime(ping["creationDate"], "%Y-%m-%dT%H:%M:%S.%fZ")

    appVersion = ping["meta/appVersion"]
    buildId = ping["meta/appBuildId"]
    locale = ping["locale"]
    os = ping["os"]
    osVersion = ping["osversion"]
    device = ping["device"]
    arch = ping["arch"]
    defaultSearch = ping["defaultSearch"]
    distributionId = ping["distributionId"]

    experiments = ping["experiments"]
    if experiments is None:
        experiments = []

    return [clientId, submissionDate, creationDate, profileDate, geoCountry, locale, os, osVersion, buildId, appVersion, device, arch, defaultSearch, distributionId, json.dumps(experiments)]

channels = ["nightly", "aurora", "beta", "release"]

start = dt.datetime.now() - dt.timedelta(1)
end = dt.datetime.now() - dt.timedelta(1)

day = start
while day <= end:
    for channel in channels:
        print "\nchannel: " + channel + ", date: " + day.strftime("%Y%m%d")

        kwargs = dict(
            doc_type="core",
            submission_date=(day.strftime("%Y%m%d"), day.strftime("%Y%m%d")),
            channel=channel,
            app="Fennec",
            fraction=1
        )

        # Grab all available source_version pings
        pings = get_pings(sc, source_version="*", **kwargs)

        subset = get_pings_properties(pings, ["meta/clientId",
                                              "meta/documentId",
                                              "meta/submissionDate",
                                              "meta/appVersion",
                                              "meta/appBuildId",
                                              "meta/geoCountry",
                                              "locale",
                                              "os",
                                              "osversion",
                                              "device",
                                              "arch",
                                              "profileDate",
                                              "creationDate",
                                              "defaultSearch",
                                              "distributionId",
                                              "experiments"])

        subset = dedupe_pings(subset)
        print "\nDe-duped pings:" + str(subset.count())
        print subset.first()

        transformed = subset.map(transform)
        print "\nTransformed pings:" + str(transformed.count())
        print transformed.first()

        s3_output = "s3n://net-mozaws-prod-us-west-2-pipeline-analysis/mobile/mobile_clients"
        s3_output += "/v1/channel=" + channel + "/submission=" + day.strftime("%Y%m%d") 
        schema = StructType([
            StructField("clientid", StringType(), False),
            StructField("submissiondate", TimestampType(), False),
            StructField("creationdate", TimestampType(), True),
            StructField("profiledate", TimestampType(), True),
            StructField("geocountry", StringType(), True),
            StructField("locale", StringType(), True),
            StructField("os", StringType(), True),
            StructField("osversion", StringType(), True),
            StructField("buildid", StringType(), True),
            StructField("appversion", StringType(), True),
            StructField("device", StringType(), True),
            StructField("arch", StringType(), True),
            StructField("defaultsearch", StringType(), True),
            StructField("distributionid", StringType(), True),
            StructField("experiments", StringType(), True)
        ])
        # Make parquet parition file size large, but not too large for s3 to handle
        coalesce = 1
        if channel == "release":
            coalesce = 4
        grouped = sqlContext.createDataFrame(transformed, schema)
        grouped.coalesce(coalesce).write.parquet(s3_output)

    day += dt.timedelta(1)

