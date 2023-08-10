#imports.... Run this each time after restarting the Kernel
#!pip install watson_developer_cloud
import watson_developer_cloud as watson
import json
from botocore.client import Config
import ibm_boto3
import requests

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

credentials_nlu = {
    "url": "",
    "username": "",
    "password": "",
    "version": "2017-02-27"
}

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

#NLU

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding.features import (
    v1 as Features)

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version = '2017-02-27',
    username = credentials_nlu['username'],
    password = credentials_nlu['password']
)

chunk_size=25 # This CHUNK size is used to disaggregate a transcript 
#e.g. in this case a 290 word transcript would have 10 chunks - 9 with 30 words and 1 with 20 words - approximates 'time domain' for this lab

def chunk_transcript(transcript, chunk_size):
    transcript = transcript.split(' ')
    return [ transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size) ] # chunking data

def process_text(text):
    transcript=''
    for sentence in json.loads(text)['results']:
        transcript = transcript + sentence['alternatives'][0]['transcript'] # concatenate sentences
    #transcript = chunk_transcript(transcript, chunk_size) # chunk the transcript
    return  transcript


def analyze_transcript(features, file_name):
    streaming_body = client.get_object(Bucket = credentials_os['BUCKET'], Key=file_name.split('.')[0]+'_text.json')['Body']
    transcript = process_text(streaming_body.read().decode("utf-8"))
    nlu_analysis = natural_language_understanding.analyze(features, transcript, return_analyzed_text=True)
    res=client.put_object(Bucket = credentials_os['BUCKET'], Key=file_name.split('.')[0]+'_NLU.json', Body= json.dumps(nlu_analysis))
    return nlu_analysis

def post_analysis(result):
    print(result['analyzed_text'])
    categories = result['categories']
    for category in categories:
        print('label: ', category['label'], ', score: ', category['score']) #add table instead of prints

        
def process_text_chunks(text):
    transcript=''
    for sentence in json.loads(text)['results']:
        transcript = transcript + sentence['alternatives'][0]['transcript'] # concatenate sentences
    transcript = chunk_transcript(transcript, chunk_size) # chunk the transcript
    return  transcript

def analyze_transcript_chunks(features, file_name):
    streaming_body = client.get_object(Bucket = credentials_os['BUCKET'], Key=file_name.split('.')[0]+'_text.json')['Body']
    transcript=streaming_body.read().decode("utf-8")
    nlu_analysis={}
    for chunk in process_text_chunks(transcript):
        chunk = ' '.join(chunk)
        print('chunk: ', chunk)
        nlu_analysis[chunk] = natural_language_understanding.analyze(features, chunk, return_analyzed_text=True, language='en')
    outfilename = file_name.split('.')[0]+'_NLUchunks.json'
    print("writing file: ", outfilename, " to cloud object storage" )
    res=client.put_object(Bucket = credentials_os['BUCKET'], Key=outfilename, Body= json.dumps(nlu_analysis))
    return nlu_analysis


def post_analysis_chunks(result):
    for chunk in result.keys():
        categories = result[chunk]['categories']
        print('\nchunk: ', chunk)
        for category in categories:
            print('label: ', category['label'], ', score: ', category['score']) #add table instead of prints

file_list = ['sample1-addresschange-positive.ogg',
             'sample2-address-negative.ogg',
             'sample3-shirt-return-weather-chitchat.ogg',
             'sample4-angryblender-sportschitchat-recovery.ogg',
             'sample5-calibration-toneandcontext.ogg',
             'jfk_1961_0525_speech_to_put_man_on_moon.ogg',
             'May 1 1969 Fred Rogers testifies before the Senate Subcommittee on Communications.ogg']

features = {"concepts":{},"entities":{},"keywords":{},"categories":{},"emotion":{},"sentiment":{},"semantic_roles":{} }

result = analyze_transcript_chunks(features, file_list[0])
post_analysis_chunks(result)

## If you'd like to execute NLU per chunk of audio file (chunk is 25 words), make sure the next lines are uncomments
for filename in file_list:
    print("\n\nprocessing file: ", filename)
    result = analyze_transcript_chunks(features,filename)
    post_analysis_chunks(result)



