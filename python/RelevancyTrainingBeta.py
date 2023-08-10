import watson_developer_cloud
import json
import csv
import requests

#create a new Discovery object using the python SDK and credentials from bluemix. 
username="INSERT CREDENTIALS HERE"
password="INSERT CREDENTIALS HERE"

discovery = watson_developer_cloud.DiscoveryV1(
    '2016-11-07',
    username=username,
    password=password)

#specify the environment and collection where the content lives. These ids can be collected from 
#the discovery web tooling collection details page.
environment = "7455992d-3f0c-4936-b4d3-1dc8d0277e50"
collection = "800ef70c-7ac9-4a77-9311-6ab96adc7751"

with open ("/bww_data/questions_train.txt") as questions:
    #open an output file to place the responses 
    filestr = "/bww_data/training_file.tsv"
    of = open(filestr, "w")
    writer = csv.writer(of, delimiter="\t")
    
    #go through each question in file and prepare Discovery query paramaters 
    for line in questions:
        question = line.replace("\n", "")
        params = {}
        params["query"] = "%s" % (question)
        params["return"] = "_id,body,title" #these fields may need to be updated depending on the content being used 
        params["count"] = 4 
        
        #run Discovery query to get results from untrained service 
        result = discovery.query(environment_id=environment, collection_id=collection, query_options=params)
        
        #create a row for each query and results 
        result_list = [question.encode("utf8")]
        for resultDoc in result["results"]:
            id = resultDoc["id"]
            body = resultDoc["body"].encode("utf8")
            title = resultDoc["title"].encode("utf8")
            result_list.extend([id,title,body,' ']) #leave a space to enter a relevance label for each doc 
        
        #write the row to the file 
        writer.writerow(result_list)
    
    of.close()

#function for posting to training data endpoint 
def training_post(discovery_path, training_obj):
    training_json = json.dumps(training_obj)
    headers = {
        'content-type': "application/json"
        }
    auth = (username, password)
    r = requests.request(method="POST",url=discovery_path,data=training_json,headers=headers,auth=auth)
 
#open the training file and create new training data objects
with open(filestr,'r') as training_doc:
    training_csv = csv.reader(training_doc, delimiter='\t')    
    training_obj = {}
    training_obj["examples"] = []
    
    discovery_path = "https://gateway.watsonplatform.net/discovery/api/v1/environments/" + environment + "/collections/" + collection 
    discovery_training_path = discovery_path + "/training_data?version=2016-11-07"
    
    #create a new object for each example 
    for row in training_csv:
        training_obj["natural_language_query"] = row[0]
        i = 1 
        for j in range(1,3):
            example_obj = {}
            example_obj["relevance"] = row[i+3]
            example_obj["document_id"] = row[i]
            training_obj["examples"].append(example_obj)
            i = i + 4 

        #send the training data to the discovery service 
        training_post(discovery_training_path, training_obj)

    

status = discovery.get_collection(environment,collection)
print(json.dumps(status))

def relevance_query(path, query):
    headers = {
        'content-type': "application/json"
        }
    params = {}
    params["natural_language_query"] = query
    params["version"] = "2016-11-07"
    params["return"] = "_id,body,title"
    params["count"] = 3
    auth = (username, password)
    r = requests.request(method="GET",url=path,params=params,headers=headers,auth=auth)
    #print(r.text) 

#replace with path to your questions 
test_questions_path = "c:/users/IBM_ADMIN/documents/data/wimbledon/wimbledon/questions/questions_test.txt" 

discovery_query_path = discovery_path + "/query"

#perform a natural_language_query 
with open(test_questions_path, 'r') as test_questions:
    for question in test_questions:
        print(discovery_query_path)
        relevance_query(discovery_query_path, question)
    



