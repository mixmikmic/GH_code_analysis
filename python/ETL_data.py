from bs4 import BeautifulSoup
import pandas as pd
import os

# parse a list of files into a python dictionary
datapath = "./transcripts/"
transcripts = {}

for file in os.listdir(datapath):
    if file.endswith(".xml"):
        soup = BeautifulSoup(open(datapath+ file), 'html.parser')
        result = [(p['begin'], p['end'], p.text) for p in soup.find_all('p')]
        transcripts[file] = result

# load dict into pandas dataframe
transcripts_pd = pd.DataFrame()
for transcript in sorted(transcripts): # may want to limit the list for convience/testing
    df2=pd.DataFrame(transcripts[transcript], columns = ['sTimestamp','eTimestamp','words'])
    #words dont always seemm to line up with the video, so rounding is implemented. lets see if this works well overall
    df2['sTime'] = pd.to_datetime(df2['sTimestamp']).dt.round('s').dt.strftime("%Hh%Mm%Ss")
    df2['videoId'] = transcript
    #take the file format off the ID. Why are we keeping this? Not sure, perhaps in order to track the file it came from in case we switch formats
    videoId_strip = transcript[:-4]
    #create the youtube permalink for sharing at the specified time
    df2['share_url'] = "https://youtu.be/" + videoId_strip + "?t=" + df2['sTime']
    transcripts_pd = transcripts_pd.append(df2)

transcripts_pd.head()

def search_string_cols(df, string, col):
    """searches specified colummn in dataframe for arbitrary string"""
    results = pd.DataFrame(df[df[col].str.contains(string, na=False)])
    return results

results = search_string_cols(transcripts_pd, 'Chicago', 'words')

results.head()

results['share_url']



