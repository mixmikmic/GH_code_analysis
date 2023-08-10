import requests
import json
import time
import os, sys
from IPython.display import display
import datetime
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
import nltk
import numpy as np
import time

# Define the absolute path of directory where the data should be stored 
data_path = "/home/gkastro/title-prediction-tensorflow/content-data/"

# Define the starting year for our search
min_year = 2008

# Create the directories/folders for the data to be stored
if not os.path.isdir(data_path+"/text-data"):
    os.mkdir(data_path+"/text-data")
if not os.path.isdir(data_path+"/vocabs"):
    os.mkdir(data_path+"/vocabs")
for year in range(min_year, 2018):
    if not os.path.isdir(data_path+"/"+str(year)):
        os.mkdir(data_path+"/"+str(year))
    for month in range(1,13):
        if not os.path.isdir(data_path+"/"+str(year)+"/"+str(month)):
            os.mkdir(data_path+"/"+str(year)+"/"+str(month))

# The S-API and the C-API keys should be stored in environment variables SAPI_key and CAPI_key
error_dir = data_path+"errors/"
start = time.time()

# We need to define the date after which we begin our search.
# Naming this variable min_date might seem more appropriate but the primary use of it is to keep track of the
# most recent date that has been fetched, while we perform requests one after the other.
max_date = str(min_year)+"-01-01T00:00:00Z"

# Define the number of iterations/requests, 100 results are brought back from each request,
# out of which some articles might not be available through C-API.
# So after performing 1000 requests we should expect to have retrieved ~95,000 articles
if "SAPI_key" in os.environ and "CAPI_key"in os.environ:
    s_api_key = os.environ["SAPI_key"]
    c_api_key = os.environ["CAPI_key"]
    for y in range(0,1000):
        headers = {'Content-Type': 'application/json'}
        payload = {"queryString":"lastPublishDateTime:>"+max_date,
                   "queryContext":{
                       "curations":["ARTICLES", "BLOGS"]
                   },
                   "resultContext":{
                       "maxResults":100, 
                       "offset":0,
                       "aspects":["title", "metadata", "lifecycle"],
                       "sortOrder":"ASC",
                       "sortField":"lastPublishDateTime"
                   } 
                  }
        r1 = requests.post("https://api.ft.com/content/search/v1?apiKey="+str(s_api_key), headers=headers, json=payload)
        # If any error occurs while performing a request we carry on with the next request
        if r1.status_code >= 400:
            continue
        response_json1 = r1.json()
        # If there is no article matching our search then we break our request-loop,
        # since we have reached the present day or no more article are available
        if response_json1["results"][0]["indexCount"] == 0:
            break
        response_json1_length = len(response_json1["results"][0]["results"])
        # Update max_date to the publish date of most recent article fetched
        max_date = response_json1["results"][0]["results"][response_json1_length-1]["lifecycle"]["lastPublishDateTime"]   
        # Iterate through the results of S-API in order to get data through the enriched content API
        for i in response_json1["results"][0]["results"]:
            if "title" in i.keys() and "id" in i.keys():
                item_id = i["id"]
                tmp = i            
                url = "https://api.ft.com/enrichedcontent/"+str(item_id)+"?apiKey="+str(c_api_key)
                r2 = requests.get(url)
                if r2.status_code >= 400:
                    continue
                response_json2 = r2.json()
                if "errors" in response_json2.keys():
                    t = open(error_dir+item_id+".json", "w")
                    json.dump({"status_code":r2.status_code, "url":r2.url, "text":r2.text}, t, indent=4)
                    t.close()
                    continue
                if "bodyXML" in response_json2.keys():
                    tmp["body"] = response_json2["bodyXML"]
                    if "prefLabel" in response_json2.keys():
                        tmp["prefLabel"] = response_json2["prefLabel"]
                    else:
                        tmp["prefLabel"] = ""
                    if "standfirst" in response_json2.keys():
                        tmp["standfirst"] = response_json2["standfirst"]
                    else:
                        tmp["standfirst"] = ""
                    dtm = datetime.datetime.strptime(i["lifecycle"]["lastPublishDateTime"], "%Y-%m-%dT%H:%M:%SZ")
                    # Saving all the data retrieved for each article in a separate json file, within a year and month folder
                    f = open(data_path+str(dtm.year)+"/"+str(dtm.month)+"/"+item_id+".json", "w")
                    json.dump(tmp, f, indent=4)
                    f.close()
                else:
                    continue
            else:
                continue
else:
    print("API keys missing !")
end = time.time()
print(end - start)

start = time.time()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Define the range of years and months of the articles that we want to transform
for year in range(2008,2018):
    for month in range(1,13):
        # We will create a vocabulary and a content file for each year and month
        content_file = data_path+"text-data/text-data-"+str(year)+"-"+str(month)
        vocab_file = data_path+"vocabs/vocab-"+str(year)+"-"+str(month)+".csv"
        file = open(content_file, "w")
        vocab_df = pd.DataFrame(columns=["words", "frequency"])
        for filename in os.listdir(str(year)+"/"+str(month)+"/"):
            if filename.endswith(".json"):
                file2 = open(str(year)+"/"+str(month)+"/"+filename, "r")
                content = json.load(file2)
                file2.close()
                title = content["title"]["title"].replace("\n", " ").replace("\r", "").replace("="," ").replace("\t", " ")
                title_tok = unicodedata.normalize("NFKD",title).encode("ascii", "ignore")
                body_raw = unicodedata.normalize("NFKD",content["body"]).encode("ascii", "ignore")
                # Getting rid of the html tags
                soup = BeautifulSoup(body_raw, "html.parser")
                soup_title = BeautifulSoup(title_tok, "html.parser")
                # Tokenize sentences and add <s></s> tags
                body_text = " </s> <s> ".join(tokenizer.tokenize(soup.get_text())).replace("\n", " ").replace("\r", "").replace("="," ").replace("\t", " ")
                body = "<d> <s> "+body_text+" </s> </d>"
                # Retrieve the tokens and create the vocabulary
                tokens = nltk.wordpunct_tokenize(soup.text+soup_title.text)
                words = [w for w in tokens]
                words_freq = [words.count(w) for w in words]
                d = {"words":words, "frequency":words_freq}
                vocab_tmp = pd.DataFrame(data=d, columns=["words", "frequency"])
                vocab_tmp.drop_duplicates(keep="first", inplace=True, subset="words")
                # If a vocabulary already exists for the given year and month then we update it
                vocab_df = pd.merge(vocab_df, vocab_tmp, how = "outer", on = "words")
                vocab_df.fillna(value=0, inplace=True)
                vocab_df["frequency"] = vocab_df.frequency_x + vocab_df.frequency_y
                vocab_df.drop(labels=["frequency_x", "frequency_y"], axis=1, inplace=True)
                file.write("abstract=<d> <p> <s> "+title+" </s> </p> </d>\tarticle= "+body+"\n")
        file.close()
        vocab_df.sort(ascending=False, columns="frequency", inplace=True)
        vocab_df.to_csv(data_path+"vocabs/"+vocab_file)
        np.savetxt(data_path+"vocabs/vocab-"+str(year)+"-"+str(month)+".txt", vocab_df.values, fmt="%s %d")
end = time.time()
print(end - start)

# We repeat a very similar process in order to create a content file and a vocabulary for each year
for year in range(2008, 2018):
    vocab_df = pd.DataFrame(columns=["words", "frequency"])
    outfile = open(data_path+"text-data/text-data-"+str(year), "w")
    for month in range(1, 13):
        vocab_tmp = pd.read_csv(data_path+"vocabs/vocab-"+str(year)+"-"+str(month)+".csv", usecols=["words", "frequency"])[["words", "frequency"]]
        vocab_df = pd.merge(vocab_df, vocab_tmp, how = "outer", on = "words")
        vocab_df.fillna(value=0, inplace=True)
        vocab_df["frequency"] = vocab_df.frequency_x + vocab_df.frequency_y
        vocab_df.drop(labels=["frequency_x", "frequency_y"], axis=1, inplace=True)
        infile = open(data_path+"text-data/text-data-"+str(year)+"-"+str(month))
        for line in infile:
            outfile.write(line)
        infile.close()
    vocab_df.sort(ascending=False, columns="frequency", inplace=True)
    vocab_df.to_csv(data_path+"vocabs/vocab-"+str(year)+".csv")
    np.savetxt(data_path+"vocabs/vocab-"+str(year)+".txt", vocab_df.values, fmt="%s %d")
    outfile.close()

# Finally we iterate again over our data to get the content file and the vocabulary for all the articles
outfile = open(data_path+"text-data/text-data", "w")
vocab_df = pd.DataFrame(columns=["words", "frequency"])
for year in range(2008, 2018):
    vocab_tmp = pd.read_csv(data_path+"vocabs/vocab-"+str(year)+".csv", usecols=["words", "frequency"])[["words", "frequency"]]
    vocab_tmp = vocab_tmp.loc[vocab_tmp["words"]!="0"]
    vocab_tmp.to_csv(data_path+"vocabs/vocab-"+str(year)+".csv")
    vocab_df = pd.merge(vocab_df, vocab_tmp, how = "outer", on = "words")
    vocab_df.fillna(value=0, inplace=True)
    vocab_df["frequency"] = vocab_df.frequency_x + vocab_df.frequency_y
    vocab_df.drop(labels=["frequency_x", "frequency_y"], axis=1, inplace=True)
    infile = open(data_path+"text-data/text-data-"+str(year))
    for line in infile:
        outfile.write(line)
    infile.close()
# We need to add the following tokens in the vocab, their frequencies are made up but shouldn't affect the model
tmp = pd.DataFrame(data={"words":["<s>", "</s>", "<PAD>","<UNK>"], "frequency":[6000000, 6000000, 3, 2000000]}, columns = ["words", "frequency"])
vocab_df = vocab_df.append(tmp, ignore_index=True)
vocab_df.sort(ascending=False, columns="frequency", inplace=True)
# Uncomment the following line in order to keep only the 300,000 most common tokens
# vocab_df = vocab_df.iloc[0:300000,:]
vocab_df.to_csv(data_path+"vocabs/vocab.csv")
np.savetxt(data_path+"vocabs/vocab", vocab_df.values, fmt="%s %d")
outfile.close()

