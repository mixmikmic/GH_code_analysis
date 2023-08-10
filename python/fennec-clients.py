import datetime as dt
import pandas as pd
import ujson as json

from moztelemetry import get_pings, get_pings_properties, get_one_ping_per_client

get_ipython().magic('pylab inline')

update_channel = "beta"
now = dt.datetime.now()
start = now - dt.timedelta(30)
end = now - dt.timedelta(1)

pings = get_pings(sc, app="Fennec", channel=update_channel,
                  submission_date=(start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
                  build_id=("20100101000000", "99999999999999"),
                  fraction=1)

subset = get_pings_properties(pings, ["clientId",
                                      "application/channel",
                                      "application/version",
                                      "meta/submissionDate",
                                      "environment/profile/creationDate",
                                      "environment/system/os/version",
                                      "environment/system/memoryMB"])

subset = subset.filter(lambda p: p["clientId"] is not None)
print subset.first()

def transform(ping):    
    clientId = ping["clientId"] # Should not be None since we filter those out

    profileDate = None
    profileDaynum = ping["environment/profile/creationDate"]
    if profileDaynum is not None:
        profileDate = (dt.date(1970, 1, 1) + dt.timedelta(int(profileDaynum))).strftime("%Y%m%d")

    submissionDate = ping["meta/submissionDate"] # Added via the ingestion process so should not be None

    channel = ping["application/channel"]
    version = ping["application/version"]
    os_version = int(ping["environment/system/os/version"])
    memory = ping["environment/system/memoryMB"]
    if memory is None:
        memory = 0
    else:
        memory = int(memory)
            
    return [clientId, channel, profileDate, submissionDate, version, os_version, memory]

transformed = get_one_ping_per_client(subset).map(transform)
print transformed.first()

grouped = pd.DataFrame(transformed.collect(), columns=["clientid", "channel", "profiledate", "submissiondate", "version", "osversion", "memory"])
get_ipython().system('mkdir -p ./output')
grouped.to_csv("./output/fennec-clients-" + update_channel + "-" + end.strftime("%Y%m%d") + ".csv", index=False)

s3_output = "s3n://net-mozaws-prod-us-west-2-pipeline-analysis/mfinkle/android-clients-" + update_channel
s3_output += "/v1/channel=" + update_channel + "/end_date=" + end.strftime("%Y%m%d") 
grouped = sqlContext.createDataFrame(transformed, ["clientid", "channel", "profiledate", "submissiondate", "version", "osversion", "memory"])
grouped.saveAsParquetFile(s3_output)

