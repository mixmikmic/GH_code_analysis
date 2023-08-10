import boto3
import IPython
from pprint import pprint

polly = boto3.client('polly', region_name='eu-west-1')

response = polly.synthesize_speech(
    Text="It is great to see you today!",
    TextType="text",
    OutputFormat="mp3",                                           
    VoiceId="Emma")

pprint (response)
     
outfile = "pollyresponse.mp3"
data = response['AudioStream'].read()

with open(outfile,'wb') as f:
     f.write(data)
IPython.display.Audio(outfile) 

response = polly.synthesize_speech(
    Text='<speak>I am fine,<break/> thank you.<break strength="x-strong"/> \
          <prosody rate="+40%">What can I do for you?</prosody></speak>',
    TextType="ssml",
    OutputFormat="mp3",                                           
    VoiceId="Emma")
     
outfile = "pollyresponse.mp3"
data = response['AudioStream'].read()

with open(outfile,'wb') as f:
     f.write(data)
IPython.display.Audio(outfile) 

response = polly.synthesize_speech(
    Text='<speak>My favorite chemical element is <sub alias="aluminium">Al</sub>, \
    but Al prefers <sub alias="magnesium">Mg</sub>.</speak>',
    TextType="ssml",
    OutputFormat="mp3",                                           
    VoiceId="Brian")
     
outfile = "pollyresponse.mp3"
data = response['AudioStream'].read()

with open(outfile,'wb') as f:
     f.write(data)
IPython.display.Audio(outfile) 

response = polly.synthesize_speech(
    Text='My favorite chemical element is Mg',
    TextType="text",
    OutputFormat="mp3",                                           
    VoiceId="Brian",
    LexiconNames=["PollyPSE"]
    )
     
outfile = "pollyresponse.mp3"
data = response['AudioStream'].read()

with open(outfile,'wb') as f:
     f.write(data)
IPython.display.Audio(outfile) 

response = polly.get_lexicon(
    Name="PollyPSE")

xmlret = response['Lexicon']['Content']
   
print (xmlret)

response = polly.synthesize_speech(
    Text="<speak><phoneme ph='bəːɱ ˈzɛksɪʃ bəˈziːʃən dˈɛ wˈeːʃːəːn dˈɛ haʁdˈn'>Beim sächsisch besiegen die weichen die harten.</phoneme></speak>",
    TextType="ssml",
    OutputFormat="mp3",                                           
    VoiceId="Hans"
    )
     
outfile = "pollyresponse.mp3"
data = response['AudioStream'].read()

with open(outfile,'wb') as f:
     f.write(data)
IPython.display.Audio(outfile) 

