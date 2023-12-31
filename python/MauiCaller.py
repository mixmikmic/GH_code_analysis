import re

from subprocess import Popen, PIPE, STDOUT
p = Popen(["java", "-jar", "-Xmx1024m", "maui-standalone-1.1-SNAPSHOT.jar", 
         "train", "-l","data/jobscan/train", "-m","data/models/keyword_extraction_model_Steph3","-v","ACMTaxonomySkosExtended2.rdf","-f","skos","-o","1"], stdout=PIPE, stderr=STDOUT)
#for line in p.stdout:
 #   print line

get_ipython().system('pwd')

from subprocess import Popen, PIPE, STDOUT
import re
p = Popen(["java", "-jar", "-Xmx1024m", "maui-standalone-1.1-SNAPSHOT.jar", 
           "test", "-l","data/jobscanManualClean/test", "-m","data/models/keyword_extraction_model_Steph2",
           "-v","ACMTaxonomySkosExtended2.rdf","-f","skos","-n","30"], stdout=PIPE, stderr=STDOUT)
for line in p.stdout:
    if line.find("MauiTopicExtractor")<>-1:
           print line

from subprocess import Popen, PIPE, STDOUT
import re
p = Popen(["java", "-jar", "-Xmx1024m", "maui-standalone-1.1-SNAPSHOT.jar", 
         "test", "-l","data/jobscanManualClean/train", "-m","data/models/keyword_extraction_model",
           "-v","ACMTaxonomySkos.rdf","-f","skos","-n","30"], stdout=PIPE, stderr=STDOUT)
results={}
for line in p.stdout:
    if line.find("MauiTopicExtractor")<>-1:
        if line.find("Processing document")<>-1:
            #open a new doc
            doc= line.split("Processing document: ")[-1].split("\n")[0]
            kw={}
        elif line.find("Topic ")<>-1:
            key=line.split("Topic ")[-1].split( " 1 ")[0]
            value=line.split( " 1 ")[-1].split(" > ")[0]
            kw[key]=value
    results[doc]=kw

from subprocess import Popen, PIPE, STDOUT
import os,time
import json

pathToMaui="/Users/steph/Google Drive/MIDS/Capstone/JobFictionLocal/job-fiction/analyze/KeywordExtraction/"
# Function to train a MAUI model. 
# Input: 
# - path to training file, 
# - suffix so we can generate our own model
# - minimum number of occurence
# Output:
# - path to model file

def trainMaui(pathToTrain, modelID, minOccurence):
    pathToModel=pathToMaui+"data/models/keyword_extraction_model_"+modelID
    p = Popen(["java", "-jar", "-Xmx1024m", pathToMaui+"maui-standalone-1.1-SNAPSHOT.jar", 
         "train", "-l",pathToTrain, "-m",pathToModel,
               "-v",pathToMaui+"ACMTaxonomySkosExtended2.rdf","-f","skos","-o",str(minOccurence)], stdout=PIPE, stderr=STDOUT)
    for line in p.stdout:
        if line.find("WARN"):
            continue
        print line
    return pathToModel


# Function to test a MAUI model. 
# Input: 
# - path to test file, 
# - modelID for differenting models (so we can work on different model)
# - maximum number of keywords to return
# Output:
# - A JSON containing the keywords

def testMaui(pathToTest, modelID, numKw):
    pathToModel=pathToMaui+"data/models/keyword_extraction_model_"+modelID
    p = Popen(["java", "-jar", "-Xmx1024m", pathToMaui+"maui-standalone-1.1-SNAPSHOT.jar", 
         "test", "-l",pathToTest, "-m",pathToModel,
           "-v",pathToMaui+"ACMTaxonomySkosExtended2.rdf","-f","skos","-n",str(numKw)], stdout=PIPE, stderr=STDOUT)
    results={}
    kw={}
    doc=""
    init=0
    
    gen=(line for line in p.stdout if line.find("MauiTopicExtractor")<>-1 )
    for line in gen:
        if line.find("Processing document")<>-1:
            #open a new doc
            doc= line.split("Processing document: ")[-1].split("\n")[0]
            kw={}
            continue

        elif line.find("Topic ")<>-1:
            key=line.split("Topic ")[-1].split( " 1 ")[0]
            
            key=key.replace("."," ")#removing . because mongodb does not like dot in keys

            value=line.split( " 1 ")[-1].split(" > ")[0]
            kw[key]=value
            init=1
        if init:
            results[doc]=kw
    return json.dumps(results)

# Function takes a JSON, save it into a directory and call the testMaui function.
# Query should be in the form of {"jobID1":"summary","jobID2":"summary2",...}
# You can choose the model. Default is Steph2
def mauiTopicClf(query,model="Steph2",thres1=0.6,thres2=0.2):
    #create a temporary directory for MAUI to work in
    if not os.path.exists("workbench"):
        os.makedirs("workbench")
    # load the query and split each document into a separate file
    data=json.loads(query)
    for k,v in data.iteritems():
        with open("workbench/"+k+".txt",'w') as recFile:
            recFile.write(v)
    # call the Maui wrapper on these files
    response= json.loads(testMaui("workbench", model, 40))
    # remove the working directory
    shutil.rmtree("./workbench")
    

    results={}
    for k,v in response.iteritems():
        key=k.split(".txt")[0]
        mustHave={}
        niceHave={}
        exclude={}
        keywords={}
        for k2,v2 in v.iteritems():    
            if float(v2)>thres1:
                mustHave[k2]=float(v2)
            elif float(v2)>thres2:
                niceHave[k2]=float(v2)
            else:
                exclude[k2]=float(v2)
            keywords['MustHave']= mustHave
            keywords['NiceHave']= niceHave
            keywords['Excluded']= exclude
        results[key]=keywords
    
    return json.dumps(results)

get_ipython().system('ls -l data/models')


# Train a Maui model
pathToTrain=pathToMaui+"data/jobscan/train"
print trainMaui(pathToTrain,"Steph3",1)

# Test the classifier
test='{"jobID1":"On your first day, we ll expect you to have: Deep understanding of big data challenges. Built solutions using Amazon Web Services, Redshift, S3, EMR, etc.Experience with Hadoop, Map/Reduce and Hive Expertise in SQL, SQL tuning, schema design, Python and ETL processesSolid understanding utilizing Web Services and Application Programming Interfaces (APIs) Experience in test automation and ensuring data quality across multiple datasets used for analytical purposes A graduate degree in Computer Science or similar discipline Commit code to open source projects Experience retrieving data from remote systems via API calls (eg REST) Experience with test automation and continuous build It s great, but not required, if you have: Experience with Tableau Have worked in a Marketing Org Have worked with Data Scientists"}'

mauiTopicClf(test)

