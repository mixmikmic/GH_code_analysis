get_ipython().system('pip install --user --upgrade pixiedust')

import pixiedust
pixiedust.enableJobMonitor()

from pyspark.sql.functions import explode, lower

# Enter your Cloudant host name
host = 'f80e969b-3c5d-4d2d-b988-ca641c17c239-bluemix.cloudant.com'
# Enter your Cloudant user name
username = 'hattentonewayserferhadyr'
# Enter your Cloudant password
password = 'd22df65fa27eb28d00d913916a82d2da96e9986a'
# Enter your source database name
database = 'cbf_chatbot_convos'

# obtain SparkSession
sparkSession = SparkSession.builder.getOrCreate()
# load data
conversation_df = sparkSession.read.format("com.cloudant.spark").    option("cloudant.host", host).    option("cloudant.username", username).    option("cloudant.password", password).    load(database)

conversation_df.printSchema()

conversation_df.count()

dialog_df = conversation_df.select(explode(conversation_df.dialogs).alias("dialog"))
dialog_df = dialog_df.select("dialog.date", 
                             lower(dialog_df.dialog.message).alias("message"),
                             "dialog.name",
                             "dialog.reply")
dialog_df.printSchema()

dialog_df.count()

display(dialog_df)

sick_dialog_df = dialog_df.filter(dialog_df.name == 'sickGetSymptoms')
sick_dialog_df.count()

known_symptom_dialog_df = dialog_df.filter((dialog_df.name == 'sickENTSymptoms') | (dialog_df.name == 'sickOtherSymptoms'))
known_symptom_dialog_df.count()

unknown_symptom_dialog_df = dialog_df.filter(dialog_df.name == 'sickUnknownSymptoms')
unknown_symptom_dialog_df.count()

sparkSession = SparkSession.builder.getOrCreate()
match_vs_miss_df = sparkSession.createDataFrame(
[('Match', known_symptom_dialog_df.count()),
 ('Miss', unknown_symptom_dialog_df.count())],
["name","count"])

display(match_vs_miss_df)

get_ipython().system('pip install stop-words')

import re
import stop_words

def parseMessage(msg):
    msg = re.sub("[^a-zA-Z ']", "", msg)
    msgWords = re.split("\s+", msg.lower())
    msgWords = filter(lambda w: w not in "", msgWords)
    stopWords = stop_words.get_stop_words('en')
    return filter(lambda w: w not in stopWords, msgWords)

rdd = unknown_symptom_dialog_df.rdd.flatMap(lambda row: parseMessage(row['message']))

rdd = rdd.map(lambda x: (x,1)).reduceByKey(lambda x, y: x + y)

rdd.take(10)

unknown_symptom_dialog_by_word_df = rdd.toDF(['word','count']).orderBy('count', ascending=False)
display(unknown_symptom_dialog_by_word_df)

display(unknown_symptom_dialog_by_word_df.limit(10))



