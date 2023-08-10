#imports.... Run this each time after restarting the Kernel
get_ipython().system('pip install watson_developer_cloud')
import watson_developer_cloud as watson
import json
from botocore.client import Config
import ibm_boto3
import requests
from urllib.request import urlopen 

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

credentials_audio_samples = {
    "apikey": "RRKwTeEGGDJtPDii9SPRxOHooiXZnUdpCRcjnzdPo3uh",
    "endpoints": "https://cos-service.bluemix.net/endpoints",
    "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:cloud-object-storage:global:a/8739a0c318b37263a932b45c1947965d:7ce353a1-fa6f-4e25-a311-e29b0b2a8ad8::",
    "iam_apikey_name": "auto-generated-apikey-3e922f8a-7453-4038-8410-5399f9306530",
    "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Reader",
    "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/8739a0c318b37263a932b45c1947965d::serviceid:ServiceId-3039af12-8feb-4ef5-8306-d6acd6a0ca31",
    "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/8739a0c318b37263a932b45c1947965d:7ce353a1-fa6f-4e25-a311-e29b0b2a8ad8::",
    "BUCKET": 'audio-samples',
}

credentials_audio_samples['BUCKET'] = 'audio-samples'

credentials_stt = {
  "url": "",
  "username": "",
  "password": ""
}

# The code was removed by DSX for sharing.

def set_up_object_storage(credentials_object_storage):
    endpoints = requests.get(credentials_object_storage['endpoints']).json()

    iam_host = (endpoints['identity-endpoints']['iam-token'])
    cos_host = (endpoints['service-endpoints']['cross-region']['us']['public']['us-geo'])

    auth_endpoint = "https://" + iam_host + "/oidc/token"
    service_endpoint = "https://" + cos_host


    client = ibm_boto3.client(
        's3',
        ibm_api_key_id = credentials_object_storage['apikey'],
        ibm_service_instance_id = credentials_object_storage['resource_instance_id'],
        ibm_auth_endpoint = auth_endpoint,
        config = Config(signature_version='oauth'),
        endpoint_url = service_endpoint
       )
    return client

client = set_up_object_storage(credentials_os)
client_global = set_up_object_storage(credentials_audio_samples)


#STT

import json
import io
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1

speech_to_text = SpeechToTextV1(
    username = credentials_stt['username'],
    password = credentials_stt['password'],
    url = 'https://stream.watsonplatform.net/speech-to-text/api',
)


# OGG, WAV FLAC, L16, MP3, MPEG formats are options for the STT service 
# with Narrowband (generaly telco) and Broadband (e.g. higher quality USB mic) audio.  
# For the LAB – OGG format was used for sample files in lab. Of other audio formats e.g. WAV - remember to change 'OGG' content_type='audio/ogg' in code below if you do.

#get transcript Very basic one
def get_transcript(audio):
    transcript = json.dumps(speech_to_text.recognize(audio=audio, content_type='audio/ogg', timestamps=True,
        word_confidence=True), indent=2)
    return transcript

def download_file(path, filename):
    url = path + filename
    print(url)
    r = requests.get(url, stream=True)
    return r.content

def analyze_sample(sample):
    streaming_body = client_global.get_object(Bucket = credentials_audio_samples['BUCKET'], Key=sample)['Body'] #http
    audio = streaming_body.read()
    text = get_transcript(audio)
    client.put_object(Bucket = credentials_os['BUCKET'], Key = sample.split('.')[0] + '_text.json', Body = text)
    return text

def visualize(transcript):
    for result in json.loads(transcript)['results']:
        print(result['alternatives'][0]['transcript'], result['alternatives'][0]['confidence'])    

# List of Audio files that are available in the read only Cloud Object Storage for lab re: credentials_audio_samples

file_list = ['sample1-addresschange-positive.ogg',
             'sample2-address-negative.ogg',
             'sample3-shirt-return-weather-chitchat.ogg',
             'sample4-angryblender-sportschitchat-recovery.ogg',
             'sample5-calibration-toneandcontext.ogg',
             'jfk_1961_0525_speech_to_put_man_on_moon.ogg',
             'May 1 1969 Fred Rogers testifies before the Senate Subcommittee on Communications.ogg']

# TRANSCRIBE – this is where STT receives the OGG files provided and returns text to TRANSCRIPT
# this is a test of ONE transcription in the list - place '0' - may take a minute
transcript = analyze_sample(file_list[0])
visualize(transcript)

for filename in file_list:
    print('Processing file: ', filename)
    transcript = analyze_sample(filename)
    visualize(transcript)



