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

    profileDate = None
    profileDaynum = ping["environment/profile/creationDate"]
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

    # Added via the ingestion process so should not be None.
    submissionDate = dt.datetime.strptime(ping["meta/submissionDate"], "%Y%m%d")

    appVersion = ping["application/version"]
    osVersion = ping["environment/system/os/version"]
    if osVersion is not None:
        osVersion = int(osVersion)
    locale = ping["environment/settings/locale"]
    
    # Truncate to 32 characters
    defaultSearch = ping["environment/settings/defaultSearchEngine"]
    if defaultSearch is not None:
        defaultSearch = defaultSearch[0:32]

    # Build up the device string, truncating like we do in 'core' ping.
    device = ping["environment/system/device/manufacturer"]
    model = ping["environment/system/device/model"]
    if device is not None and model is not None:
        device = device[0:12] + "-" + model[0:19]

    xpcomABI = ping["application/xpcomAbi"]
    arch = "arm"
    if xpcomABI is not None and "x86" in xpcomABI:
        arch = "x86"

    return [clientId, profileDate, submissionDate, creationDate, appVersion, osVersion, locale, defaultSearch, device, arch]

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
                                              "creationDate",
                                              "application/version",
                                              "environment/system/os/version",
                                              "environment/profile/creationDate",
                                              "environment/settings/locale",
                                              "environment/settings/defaultSearchEngine",
                                              "environment/system/device/model",
                                              "environment/system/device/manufacturer",
                                              "application/xpcomAbi"])

        subset = dedupe_pings(subset)
        print "\nDe-duped pings:"
        print subset.first()

        transformed = subset.map(transform)
        print "\nTransformed pings:"
        print transformed.first()

        s3_output = "s3n://net-mozaws-prod-us-west-2-pipeline-analysis/mobile/android_clients"
        s3_output += "/v1/channel=" + channel + "/submission=" + day.strftime("%Y%m%d") 
        schema = StructType([
            StructField("clientid", StringType(), False),
            StructField("profiledate", TimestampType(), True),
            StructField("submissiondate", TimestampType(), False),
            StructField("creationdate", TimestampType(), True),
            StructField("appversion", StringType(), True),
            StructField("osversion", IntegerType(), True),
            StructField("locale", StringType(), True),
            StructField("defaultsearch", StringType(), True),
            StructField("device", StringType(), True),
            StructField("arch", StringType(), True)
        ])
        grouped = sqlContext.createDataFrame(transformed, schema)
        grouped.coalesce(1).write.parquet(s3_output)

    day += dt.timedelta(1)

