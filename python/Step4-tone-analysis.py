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

# Add your TONE Service credentials here
credentials_tone = {
    "url": '',
    "username": '',
    "password": ''
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



from watson_developer_cloud import ToneAnalyzerV3

tone_analyzer = ToneAnalyzerV3(version = '2016-05-19',
                               username = credentials_tone['username'],
                               password = credentials_tone['password'])


chunk_size=25

def chunk_transcript(transcript, chunk_size):
    transcript = transcript.split(' ')
    return [ transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size) ] # chunking data
    

def process_text(text):
    transcript=''
    for sentence in json.loads(text)['results']:
        transcript = transcript + sentence['alternatives'][0]['transcript'] # concatenate sentences
    transcript = chunk_transcript(transcript, chunk_size) # chunk the transcript
    return transcript


def analyze_transcript(file_name):
    transcript = client.get_object(Bucket = credentials_os['BUCKET'], Key = file_name.split('.')[0]+'_text.json')['Body']
    transcript = transcript.read().decode("utf-8")
    tone_analysis={}
    for chunk in process_text(transcript):
        if len(chunk) > 2:
            chunk = ' '.join(chunk)
            tone_analysis[chunk] = tone_analyzer.tone(chunk, content_type='text/plain')
    res=client.put_object(Bucket = credentials_os['BUCKET'], Key= file_name.split('.')[0]+'_tone.json', Body = json.dumps(tone_analysis))
    return tone_analysis

def print_tones(tones):
    for tone in tones:
        print(tone) ## note for self: update this and show table instead

def post_analysis(result):
    for chunk in result.keys():
        tone_categories = result[chunk]['document_tone']['tone_categories']
        print('\nchunk: ', chunk)
        for tone_category in tone_categories:
            print_tones(tone_category['tones']) #add table instead of prints

file_list = ['sample1-addresschange-positive.ogg',
             'sample2-address-negative.ogg',
             'sample3-shirt-return-weather-chitchat.ogg',
             'sample4-angryblender-sportschitchat-recovery.ogg',
             'sample5-calibration-toneandcontext.ogg',
             'jfk_1961_0525_speech_to_put_man_on_moon.ogg',
             'May 1 1969 Fred Rogers testifies before the Senate Subcommittee on Communications.ogg']

result = analyze_transcript(file_list[0])
post_analysis(result) ## aggregrate tones then show histogram and then filter 

for filename in file_list:
    print("\n\nprocessing file: ", filename)
    result = analyze_transcript(filename)
    post_analysis(result) ## aggregrate tones then show histogram and then filter 



