# Run pip install only the first time, once installed on your Spark machine, no need to re-run unless you want to upgrade
get_ipython().system('pip install --upgrade --force-reinstall wordcloud')
get_ipython().system('pip install --user --upgrade pixiedust')

import ibm_boto3
from botocore.client import Config
import json
import pixiedust
from pixiedust.display import *

import requests

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.functions import col

# For Cloud Object Storage - populate your own information here from "SERVICES" on this page, or Console Dashboard on ibm.com/cloud

# From service dashboard page select Service Credentials from left navigation menu item
credentials_os = {
  "apikey": "",
  "cos_hmac_keys": {
    "access_key_id": "",
    "secret_access_key": ""
  },
  "endpoints": "https://cos-service.bluemix.net/endpoints",
  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance",
  "iam_apikey_name": "",
  "iam_role_crn": "",
  "iam_serviceid_crn": "",
  "resource_instance_id": ""
}

# Buckets are created for you when you create project. From service dashboard page select Buckets from left navigation menu item, 
credentials_os['BUCKET'] = '<bucket_name>' # copy bucket name from COS

# The code was removed by DSX for sharing.

endpoints = requests.get(credentials_os['endpoints']).json()

iam_host = (endpoints['identity-endpoints']['iam-token'])
cos_host = (endpoints['service-endpoints']['cross-region']['us']['public']['us-geo'])

auth_endpoint = "https://" + iam_host + "/oidc/token"
service_endpoint = "https://" + cos_host


client = ibm_boto3.client(
    's3',
    ibm_api_key_id = credentials_os['apikey'],
    ibm_service_instance_id = credentials_os['resource_instance_id'],
    ibm_auth_endpoint = auth_endpoint,
    config = Config(signature_version='oauth'),
    endpoint_url = service_endpoint
   )

# Method to parse NLU response file from Cloud Object Storage
# and return sentiment score, sentiment label, and keywords
# This method works for the scenario of one NLU call per call (file)
def getNLUresponse(COSclient, bucket, files):
    nlu_results = []
    for filename in files:
        # Extract NLU enriched filename from the original file name
        nlu_filename = filename.split('.')[0]+'_NLU.json'
        print("Processing NLU response from file: ", nlu_filename)
        streaming_body = COSclient.get_object(Bucket=bucket, Key=nlu_filename)['Body']
        nlu_response = json.loads(streaming_body.read().decode("utf-8"))
        #print(json.dumps(nlu_response,indent=2))
        if nlu_response and nlu_response['sentiment']         and nlu_response['sentiment']['document'] and nlu_response['sentiment']['document']['label']:
            sentiment_score = nlu_response['sentiment']['document']['score']
            sentiment_label = nlu_response['sentiment']['document']['label']
            keywords = list(nlu_response['keywords'])
        else:
            sentiment_score = 0.0
            sentiment_label = None
            keywords = null
        nlu_results.append((filename,sentiment_score,sentiment_label,keywords))
    return (nlu_results)


# Method to parse NLU Emotion Tone response file from Cloud Object Storage
def getChunkNLU(nlu_response):
    #print(json.dumps(nlu_response,indent=2))
    if nlu_response and nlu_response['sentiment']     and nlu_response['sentiment']['document'] and nlu_response['sentiment']['document']['label']:
        sentiment_score = nlu_response['sentiment']['document']['score']
        sentiment_label = nlu_response['sentiment']['document']['label']
        keywords = list(nlu_response['keywords'])
    else:
        sentiment_score = 0.0
        sentiment_label = None
        keywords = null
    
    return sentiment_score, sentiment_label, keywords

# Method to parse NLU response file from Cloud Object Storage
# and return sentiment score, sentiment label, and keywords
# This method handles the scenario when call is broken into multiple chunks
def getNLUresponseChunks(COSclient, bucket, files):
    nlu_results = []
    print("files: ", files)
    for filename in files:
        # Extract NLU enriched filename from the original file name
        nlu_filename = filename.split('.')[0]+'_NLUchunks.json'
        print("Processing NLU response from file: ", nlu_filename)
        streaming_body = COSclient.get_object(Bucket=bucket, Key=nlu_filename)['Body']
        nlu_chunks_response = json.loads(streaming_body.read().decode("utf-8"))
        if nlu_chunks_response and len(nlu_chunks_response)>0:
            chunkidx = 0
            for chunk in nlu_chunks_response:
                chunk_nlu = getChunkNLU(nlu_chunks_response[chunk])
                print('chunk nlu: ', chunk_nlu)
                print('type of chunk nlu: ', type(chunk_nlu))
                chunkidx = chunkidx + 1
                tmp_results = (filename, chunkidx, chunk_nlu)
                print('tmp results: ', tmp_results)
                print('length of tmp results: ', len(tmp_results))
                l = list((filename,chunkidx)) + list(chunk_nlu)
                print('len of l: ', len(l))
                nlu_results.append(l)
               # nlu_results.append((filename, chunkidx, chunk_nlu))
        
    return (nlu_results)

# List of files which were transcribed by STT and enriched with NLU
file_list = ['sample1-addresschange-positive.ogg',
             'sample2-address-negative.ogg',
             'sample3-shirt-return-weather-chitchat.ogg',
             'sample4-angryblender-sportschitchat-recovery.ogg',
             'sample5-calibration-toneandcontext.ogg',
             'jfk_1961_0525_speech_to_put_man_on_moon.ogg',
             'May 1 1969 Fred Rogers testifies before the Senate Subcommittee on Communications.ogg']

# Specify the bucket which contains the enriched NLU files
bucket = credentials_os['BUCKET']

# Define header to map to extracted NLU features
#nlu_header=['filename','sentiment_score','sentiment_label','keywords']
#nlu_results = getNLUresponse(client,bucket,file_list)

## Alternative call to handle the case when the NLU response has been broken into chunks of 25 words each
nlu_header=['filename','chunkidx','sentiment_score','sentiment_label','keywords']
nlu_results = getNLUresponseChunks(client,bucket,file_list)
    

callcenterlogs_nluDF = spark.createDataFrame(nlu_results, nlu_header)

# Common validation calls to better understand your data
callcenterlogs_nluDF.printSchema()
callcenterlogs_nluDF.show()

## Ignore any records with null sentiment label
callcenterlogs_nluDF = callcenterlogs_nluDF.where(col('sentiment_label').isNotNull())
perlabel_sentimentDF = callcenterlogs_nluDF.groupBy('sentiment_label')                              .agg(F.count('filename')                              .alias('num_calls'))

## Take a look
perlabel_sentimentDF.show()

# Call Pixiedust to visualize sentiment data
display(callcenterlogs_nluDF)

from pyspark.sql.functions import explode

# Explode keywords
callcenterlogs_nluDF = callcenterlogs_nluDF.select(explode('keywords').alias('topkeywords'))
callcenterlogs_nluDF = callcenterlogs_nluDF.select('topkeywords').rdd.map(lambda row: row[0]).toDF()

# check top rows
callcenterlogs_nluDF.head(4)

# UDF to return lower case of word
def toLowerCase(word):
    return word.lower()


# Process extracted keywords to change to lower case
udfLowerCase = udf(toLowerCase, StringType())
callcenterlogsTopKeywordsDF = callcenterlogs_nluDF.withColumn('topkeywords',udfLowerCase('text'))

# Group by topkeywords and compute average relevance per keyword and also number of calls for each keyword
callcenterlogsKwdsNumDF = callcenterlogsTopKeywordsDF.groupBy('topkeywords')                              .agg(F.count('topkeywords').alias('kwdsnumcalls'))
callcenterlogsKwdsRelDF = callcenterlogsTopKeywordsDF.groupBy('topkeywords')                          .agg(F.avg('relevance').alias('kwdsavgrelevance'))

# join the keywords nunber and keywords relevance dataframes into one
callcenterlogsKeywordsDF = callcenterlogsKwdsNumDF.join(callcenterlogsKwdsRelDF,'topkeywords','outer')

# Define keyword score as product of number of calls expressing that keyword and average relevance of that keyword
callcenterlogsKeywordsDF = callcenterlogsKeywordsDF.withColumn('keyword_score',callcenterlogsKeywordsDF.kwdsnumcalls * callcenterlogsKeywordsDF.kwdsavgrelevance)

# Sort dataframe in descending order of KEYWORD_SCORE
callcenterlogsKeywordsDF = callcenterlogsKeywordsDF.orderBy('keyword_score',ascending=False)

# Remove None keywords
callcenterlogsKeywordsDF = callcenterlogsKeywordsDF.where(col('topkeywords').isNotNull())

print("Top Keywords from call center logs")
callcenterlogsKeywordsDF.show()

# visualize top keywords with pixiedust
display(callcenterlogsKeywordsDF)

# Map to Pandas DataFrame
callcenterlogsKeywordsPandas = callcenterlogsKeywordsDF.toPandas()

from wordcloud import WordCloud

# Process Pandas DataFrame in the right format to leverage wordcloud.py for plotting
# See documentation: https://github.com/amueller/word_cloud/blob/master/wordcloud/wordcloud.py 
def prepForWordCloud(pandasDF,n):
    kwdList = pandasDF['topkeywords']
    sizeList = pandasDF['keyword_score']
    kwdSize = {}
    for i in range(n):
        kwd=kwdList[i]
        size=sizeList[i]
        kwdSize[kwd] = size
    return kwdSize

get_ipython().run_line_magic('matplotlib', 'inline')
maxWords = len(callcenterlogsKeywordsPandas)
nWords = 20

#Generating wordcloud. Relative scaling value is to adjust the importance of a frequency word.
#See documentation: https://github.com/amueller/word_cloud/blob/master/wordcloud/wordcloud.py
callcenterlogsKwdFreq = prepForWordCloud(callcenterlogsKeywordsPandas,nWords)
callcenterlogsWordCloud = WordCloud(max_words=maxWords,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(callcenterlogsKwdFreq)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15,15))
ax.imshow(callcenterlogsWordCloud)

# turn off axis and ticks
plt.axis("off")


plt.show()

# Method to parse NLU Emotion Tone response file from Cloud Object Storage
def getChunkTone(tone_categories):
    for category in tone_categories:
        if category['category_id'] == 'emotion_tone':
            tones = category['tones']
    return (tones)

def getTAresponse(COSclient, bucket, files):
    tone_results=[]
    for filename in files:        
        tone_filename = filename.split('.')[0]+'_tone.json'
        print("Processing Tone Analyzer response from file: ", tone_filename)
        streaming_body = COSclient.get_object(Bucket=bucket, Key=tone_filename)['Body']
        ta_response = json.loads(streaming_body.read().decode("utf-8"))
        if ta_response and len(ta_response)>0:
            chunkidx=0
            for chunk in ta_response:
                chunk_tones = getChunkTone(ta_response[chunk]['document_tone']['tone_categories'])
                chunkidx = chunkidx + 1
                tone_results.append((filename, chunkidx, chunk_tones))
    return (tone_results)

# original list of audio files
file_list = ['sample1-addresschange-positive.ogg',
             'sample2-address-negative.ogg',
             'sample3-shirt-return-weather-chitchat.ogg',
             'sample4-angryblender-sportschitchat-recovery.ogg',
             'sample5-calibration-toneandcontext.ogg',
             'jfk_1961_0525_speech_to_put_man_on_moon.ogg',
             'May 1 1969 Fred Rogers testifies before the Senate Subcommittee on Communications.ogg']

# Get the emotion tones for all the audio files
ta_header=['filename','chunkindex','tones']
ta_results=getTAresponse(client,bucket,file_list)

# Create a Spark dataframe based on the extracted emotion tones
callcenterlogs_taDF = spark.createDataFrame(ta_results, ta_header)

# Print top rows
callcenterlogs_taDF.head(4)

callcenterlogs_taDF.printSchema()

# If not imported earlier, import explode
from pyspark.sql.functions import explode

# Explode tones
callcenterlogs_taDF_tones = callcenterlogs_taDF.select(explode('tones').alias('toptones'))
callcenterlogs_taDF_tones = callcenterlogs_taDF_tones.select('toptones').rdd.map(lambda row: row[0]).toDF()

# Print schema and note that score is of type string
callcenterlogs_taDF_tones.printSchema()

# Cast the score column from String to Double
callcenterlogs_taDF_tones = callcenterlogs_taDF_tones.withColumn("score", col("score").cast("double"))

# Print schema to verify score is now of type double
callcenterlogs_taDF_tones.printSchema()

callcenterlogs_taDF_tones.head(5)

callcenterlogs_taDF.show()

callcenterlogs_taDF.printSchema()

# If not imported earlier, import explode
from pyspark.sql.functions import explode

# Explode tones
callcenterlogs_taDF = callcenterlogs_taDF.withColumn('toptones',explode('tones'))

callcenterlogs_taDF.printSchema()

callcenterlogs_taDF.head(5)

# Select only the columns of interest
callcenterlogs_taDF_v1 = callcenterlogs_taDF.select('filename','chunkindex','toptones')

callcenterlogs_taDF_v1.head(6)

# Flatten nested fields
callcenterlogs_taDF_v2 = callcenterlogs_taDF_v1.select(F.col("filename").alias("filename"),F.col("chunkindex").alias("chunkindex"), F.col("toptones.tone_name").alias("tone_name"), F.col("toptones.tone_id").alias("tone_id"), F.col("toptones.score").alias("score"))

callcenterlogs_taDF_v2.printSchema()

# Cast the score column from String to Double
callcenterlogs_taDF_v2 = callcenterlogs_taDF_v2.withColumn("score", col("score").cast("double"))

callcenterlogs_taDF_v2.show()

# Group by toptones and compute average score per tone and also number of calls for each tone
callcenterlogsTonesNumDF = callcenterlogs_taDF_v2.groupBy('tone_id')                           .agg(F.count('tone_id').alias('tonesnumcalls'))
callcenterlogsTonesScoreDF = callcenterlogs_taDF_v2.groupBy('tone_id')                          .agg(F.avg('score').alias('tonesavgscore'))
    

# join the tones nunber and tones scores dataframes into one
callcenterlogsTonesDF = callcenterlogsTonesNumDF.join(callcenterlogsTonesScoreDF,'tone_id','outer')

# Define tones score as product of number of calls expressing that tone and average score of that tone
callcenterlogsTonesDF = callcenterlogsTonesDF.withColumn('tones_score',callcenterlogsTonesDF.tonesnumcalls * callcenterlogsTonesDF.tonesavgscore)

# Sort dataframe in descending order of tones_score
callcenterlogsTonesDF = callcenterlogsTonesDF.orderBy('tones_score',ascending=False)

# Remove None tones
callcenterlogsTonesDF = callcenterlogsTonesDF.where(col('tone_id').isNotNull())

callcenterlogsTonesDF.show()

# Translate spark dataframe into Pandas dataframe for plotting
callcenterlogsTonesPandas = callcenterlogsTonesDF.toPandas()

import numpy as np

tone_labels = callcenterlogsTonesPandas['tone_id']
tone_values = callcenterlogsTonesPandas['tonesavgscore']
xindices = np.arange(len(tone_values))

m = tone_values.max()
start=0.0
stop=m+0.2
step=0.1
yindices = np.arange(start,stop,step)

## Plot bar chart of top tones aggregated across all calls
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))

axes.set_xticks(xindices)
axes.set_xticklabels(tone_labels)
axes.set_xlabel('Tones')
axes.set_yticks(yindices)
axes.set_ylabel('Average Tone Score')
#axes.bar(xindices, tone_values, align='center', alpha=0.5)
axes.bar(xindices, tone_values)

axes.set_title('Emotion Tone aggregated over all calls')

plt.show()

# Print the callcenterlogs_taDF_v2 which included the expanded tones per file and chunk
callcenterlogs_taDF_v2.show()

# Next we filter the data to return only the tones for the reference file name we're interested in
filename = file_list[6]
#filename = "'" + filename + "'"
print('filtering by filename: ', filename)
callcenterlogs_taDF_v3 = callcenterlogs_taDF_v2.where(col('filename') == filename)

# verify how many records exist
callcenterlogs_taDF_v3.count()

callcenterlogs_taDF_v3.show()

# Filter the tones per tone_id so we can plot each tone separately
callcenterlogs_taDF_anger = callcenterlogs_taDF_v3.where(col('tone_id') == 'anger')
callcenterlogs_taDF_disgust = callcenterlogs_taDF_v3.where(col('tone_id') == 'disgust')
callcenterlogs_taDF_fear = callcenterlogs_taDF_v3.where(col('tone_id') == 'fear')
callcenterlogs_taDF_joy = callcenterlogs_taDF_v3.where(col('tone_id') == 'joy')
callcenterlogs_taDF_sadness = callcenterlogs_taDF_v3.where(col('tone_id') == 'sadness')

callcenterlogs_taDF_anger.show()

# convert spark dataframe to pandas for plotting
callcenterlogs_taDF_anger = callcenterlogs_taDF_anger.toPandas()
callcenterlogs_taDF_disgust = callcenterlogs_taDF_disgust.toPandas()
callcenterlogs_taDF_fear = callcenterlogs_taDF_fear.toPandas()
callcenterlogs_taDF_joy = callcenterlogs_taDF_joy.toPandas()
callcenterlogs_taDF_sadness = callcenterlogs_taDF_sadness.toPandas()

# Prepare the data for plotting
x = callcenterlogs_taDF_anger['chunkindex']
anger_tone = callcenterlogs_taDF_anger['score']
disgust_tone = callcenterlogs_taDF_disgust['score']
fear_tone = callcenterlogs_taDF_fear['score']
joy_tone = callcenterlogs_taDF_joy['score']
sadness_tone = callcenterlogs_taDF_sadness['score']

# Plot line chart for the different tones
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 30))
axes.plot(x, anger_tone, linewidth=2, color='purple')
axes.plot(x, disgust_tone, linewidth=2, color='yellow')
axes.plot(x, fear_tone, linewidth=2, color='red')
axes.plot(x, joy_tone, linewidth=2, color='blue')
axes.plot(x, sadness_tone, linewidth=2, color='green')

axes.set_xticks(x.index.tolist())

axes.set_xlabel('Chunk index')
axes.set_ylabel('Tone score')
axes.set_title('Tone variation over time')
axes.legend(loc="upper right", labels=['anger','disgust','fear','joy','sadness'])
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
width=0.3
axes.bar(x, anger_tone, width, color='blue', align='center', alpha=0.5)
axes.bar(x, disgust_tone, width, color='red', align='center', alpha=0.5, bottom=anger_tone)
axes.bar(x, fear_tone, width, color='green', align='center', alpha=0.5, bottom=(anger_tone + disgust_tone))
axes.bar(x, joy_tone, width, color='yellow', align='center', alpha=0.5, bottom=(anger_tone + disgust_tone + fear_tone))
axes.bar(x, sadness_tone, width, color='purple', align='center', alpha=0.5, bottom=(anger_tone + disgust_tone + fear_tone + joy_tone))

axes.set_xticks(x.index.tolist())

axes.set_xlabel('Time (Chunk index)')
axes.set_ylabel('Emotion Tone score')
axes.set_title('Emotion Tone variation over time')
axes.legend(loc="upper right", labels=['anger','disgust','fear','joy','sadness'])

plt.show()

perlabelSentimentDF = perlabel_sentimentDF.toPandas()

sentiment_labels = perlabelSentimentDF['sentiment_label']
sentiment_values = perlabelSentimentDF['num_calls']
sentiment_colors = ['green', 'gray', 'red']

get_ipython().run_line_magic('matplotlib', 'inline')
maxWords = len(callcenterlogsKeywordsPandas)
nWords = 15

#Generating wordcloud. Relative scaling value is to adjust the importance of a frequency word.
#See documentation: https://github.com/amueller/word_cloud/blob/master/wordcloud/wordcloud.py
# These variables should be computed already earlier in the notebook. If not, uncomment the next two
# lines and re-run
##callcenterlogsKwdFreq = prepForWordCloud(callcenterlogsKeywordsPandas,nWords)
##callcenterlogsWordCloud = WordCloud(max_words=maxWords,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(callcenterlogsKwdFreq)


# Create a 2x2 dashboard with 4 plots
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (30, 30))

## Set titles for images
ax[0,0].set_title('Top Keywords (aggregated across call logs)')
ax[0,1].set_title('Overall Sentiment (aggregated across call logs)')
ax[1,0].set_title('Top Emotion Tones(aggregated across call logs)')
ax[1,1].set_title('Emotion Tone Variation over Time (largest sample call log)')

                
## Plot word cloud of top keywords aggregated over all calls
ax[0,0].imshow(callcenterlogsWordCloud)

## Plot pie chart of the sentiment aggregated across all calls
ax[0,1].pie(sentiment_values, labels = sentiment_labels, colors = sentiment_colors, autopct = '%1.1f%%')

## Plot bar chart of top tones aggregated across all calls
ax[1,0].set_xticks(xindices)
ax[1,0].set_xticklabels(tone_labels)
ax[1,0].set_xlabel('Tones')
ax[1,0].set_yticks(yindices)
ax[1,0].set_ylabel('Average Score')
ax[1,0].bar(xindices, tone_values, align='center', alpha=0.5)

# Plot stacked bar chart for emotion tones and how they vary over the duration of one specific sample audio file
width=0.3
ax[1,1].bar(x, anger_tone, width, color='blue', align='center', alpha=0.5)
ax[1,1].bar(x, disgust_tone, width, color='red', align='center', alpha=0.5, bottom=anger_tone)
ax[1,1].bar(x, fear_tone, width, color='green', align='center', alpha=0.5, bottom=(anger_tone + disgust_tone))
ax[1,1].bar(x, joy_tone, width, color='yellow', align='center', alpha=0.5, bottom=(anger_tone + disgust_tone + fear_tone))
ax[1,1].bar(x, sadness_tone, width, color='purple', align='center', alpha=0.5, bottom=(anger_tone + disgust_tone + fear_tone + joy_tone))
ax[1,1].legend(loc="upper right", labels=['anger','disgust','fear','joy','sadness'])


plt.show()



