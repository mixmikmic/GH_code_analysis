import urllib.request
import json
from collections import Counter
from collections import defaultdict
from IPython.core.display import HTML
import d3_lib
from cfuzzyset import cFuzzySet as FuzzySet

#list the CIAT datasets
apiKey = "key here"
dataverseInst = "CIAT"
url = "https://dataverse.harvard.edu/api/search?q="+dataverseInst+"&key="+apiKey+"&per_page=1000&type=dataset"
r = urllib.request.urlopen(url)
data = json.loads(r.read().decode(r.info().get_param('charset') or 'utf-8'))
print(json.dumps(data, indent=2))

#foreach dataset, print the full metadata
datasets = data["data"]["items"]
bioversityDataset = []
apiKey = "key here"

for dataset in datasets:
    datasetID = dataset["global_id"]
    #print(datasetID)
    urlDataset = "https://dataverse.harvard.edu/api/datasets/export?key="+apiKey+"&exporter=dataverse_json&persistentId="+datasetID
    rDataset = urllib.request.urlopen(urlDataset)
    datasetInfo = json.loads(rDataset.read().decode(rDataset.info().get_param('charset') or 'utf-8'))
    bioversityDataset.append(datasetInfo)
    #print(json.dumps(datasetInfo, indent=2))

#save the metadata in a file - long to run
with open('CIATDataset.json', 'w') as outfile:
    json.dump(bioversityDataset, outfile)

#read the metadata dump and build the variables that will be printed

with open('CIATDataset.json') as data_file:    
    metadata = json.load(data_file)
    authors = []
    contributors = []
    keywords = []
    subjects = []
    geospatial = []
    fileTypes = []
    licenses = []
    CRPs = []
    kindsOfData = []
    titles = []
    datasetNumber = 0
    dates = []
    
    for met in metadata:
        docName = "";
        datasetNumber += 1
        #print(met["persistentUrl"])
        for tab in met["datasetVersion"]["metadataBlocks"]["citation"]["fields"]:
            if(tab["typeName"]=="title"):
                name = tab["value"]
                docName = name
                titles.append({"name":name, "docName": docName})
                
            elif(tab["typeName"]=="author"):
                for author in tab["value"]:
                    name = author["authorName"]["value"]
                    if("authorAffiliation" in author):
                        affiliation = author["authorAffiliation"]["value"]
                    else:
                        affiliation = ""

                    authors.append({"name":name, "affiliation":affiliation})      
                
            elif(tab["typeName"]=="keyword"):
                for keyword in tab["value"]:
                    name = keyword["keywordValue"]["value"]
                    
                    if '*' in name:
                        words = name.split("*")
                        for word in words:
                            word = word.strip()
                            keywords.append({"name":word})
                    elif ',' in name:
                        words = name.split(",")
                        for word in words:
                            word = word.strip()
                            keywords.append({"name":word})
                    elif ';' in name:
                        words = name.split(";")
                        for word in words:
                            word = word.strip()
                            keywords.append({"name":word})
                    else:
                        keywords.append({"name":name})
                        
                        
            elif(tab["typeName"]=="subject"):
                for subject in tab["value"]:
                    name = subject
                    subjects.append({"name":name})
                    
            elif(tab["typeName"]=="contributor"):
                for contributor in tab["value"]:
                    if("contributorName" in contributor):
                        name = contributor["contributorName"]["value"]
                        contributors.append({"name":name})
                    
            elif(tab["typeName"]=="otherReferences"):
                for CRP in tab["value"]:
                    name = CRP
                    CRPs.append({"name":name})
            
            elif(tab["typeName"]=="kindOfData"):
                for kindOfData in tab["value"]:
                    name = kindOfData
                    kindsOfData.append({"name":name})
            elif(tab["typeName"]=="dateOfDeposit"):
                name = tab["value"]
                dates.append({"name":name, "docName": docName})
            
                    
        if("geospatial" in met["datasetVersion"]["metadataBlocks"]):
            for tab in met["datasetVersion"]["metadataBlocks"]["geospatial"]["fields"]:
                if(tab["typeName"]=="geographicCoverage"):
                    for geo in tab["value"]:
                        if("otherGeographicCoverage" in geo):
                            coverage = geo["otherGeographicCoverage"]["value"].split(",")
                            for country in coverage:
                                geospatial.append({'name':country.strip()})
                        elif("country") in geo:
                            coverage = geo["country"]["value"].split(",")
                            for country in coverage:
                                geospatial.append({'name':country.strip()})
                            
        for tab in met["datasetVersion"]["files"]:
            contentType = tab["dataFile"]["contentType"]
            fileTypes.append({'name':contentType,"docName": docName})
            
                
        license = met["datasetVersion"]["license"]
        licenses.append({'name':license, "docName": docName})
    
    #print(CRPs)
    
    #print (licenses)
    #c = Counter(title['name'] for title in titles)
    #c = Counter(subject['name'] for subject in subjects)
    #c = Counter(keyword['name'] for keyword in keywords)
    #c = Counter(author['name'] for author in authors)
    #c = Counter(author['affiliation'] for author in authors)
    #c = Counter(contributor['name'] for contributor in contributors)
    #c = Counter(CRP['name'] for CRP in CRPs)
    c = Counter(geo['name'] for geo in geospatial)
    #c = Counter(fileType['name'] for fileType in fileTypes)
    #c = Counter(fileType['docName'] for fileType in fileTypes)
    #c = Counter(date['name'] for date in dates)
   
    #c = Counter(license['name'] for license in licenses)
    #result = [{'name':key, 'value':value} for key,value in c.items()]
    #c = Counter(kindOfData['name'] for kindOfData in kindsOfData)
    result = [{'name':key, 'value':value} for key,value in c.items()]
    
    print(result)
    
    #for the number of document per datasets
    #c1 = Counter(res['value'] for res in result)
    #result1 = [{'name':str(key) + " file", 'value':value} for key,value in c1.items()]
    #print(result1)
    
    #d = {}
    #for k in fileTypes:
    #    d.setdefault(k.get('name'), []).append(k.get('docName'))
    
    #for elem in result:
    #    for e in d:
    #       if(elem['name'] == e):
    #           elem['docName'] = d[e]
   
   
    #print(datasetNumber)

#### preparation for the agrovoc mapping
agrovocSimple = []
with open('agrovocLabels.json') as data_file:
    agrovoc = json.load(data_file)
    results = agrovoc["results"]["bindings"]
    for entry in results:
        uri = entry["uri"]["value"]
        label = entry["label"]["value"]
        #agrovocSimple.append({"uri": uri , "name": label})
        agrovocSimple.append(label)

#print(agrovocSimple)


####instatiation of the fuzzyset for the mappings
a=FuzzySet() # allocate the FuzzySet object

for e in agrovocSimple:
    a.add(e)

#read the metadata dump and build the variables that will be printed

with open('CIATDataset.json') as data_file:    
    metadata = json.load(data_file)
    authors = []
    contributors = []
    keywords = []
    subjects = []
    geospatial = []
    fileTypes = []
    licenses = []
    CRPs = []
    kindsOfData = []
    titles = []
    datasetNumber = 0
    dates = []
    
    ##compliance with cgCore
    titleNumber = 0
    creatorNumber = 0
    #creatorAffiliationNumber = 0
    subjectNumber = 0
    keywordNumber = 0
    descriptionNumber = 0
    publisherNumber = 0
    contributorNumber = 0
    dateNumber = 0
    typeNumber = 0
    #formatNumber  = 0
    idNumber = 0
    #sourceNumber
    #languageNumber
    #relationNumber
    geoNumber = 0 #coverage
    rightNumber = 0
    contactNumber = 0
    
    ### doc name and IDs
    publisherFilledIn = []
    
    ##donut data
    donut = {}
    donutDataset = {}
    
    
    
    for met in metadata:
        docName = "";
        datasetNumber += 1
        
        #protocol = met["protocol"]
        #authority = met["authority"]
        datasetID = met["persistentUrl"]
        
        ##per dataset
        titlePresence = 0
        creatorPresence = 0
        #creatorAffiliationPresence = 0
        subjectPresence = 0
        keywordPresence = 0
        descriptionPresence = 0
        publisherPresence = 0
        contributorPresence = 0
        datePresence = 0
        typePresence = 0
        #formatPresence  = 0
        idPresence = 0
        #sourceNumber
        #languageNumber
        #relationNumber
        geoPresence = 0 #coverage
        rightPresence = 0
        contactPresence = 0
        
        ### to know the proportion of keywords mapping to agrovoc
        keywordNumberPerDataset = 0
        keywordMatch = 0
        
        
        for tab in met["datasetVersion"]["metadataBlocks"]["citation"]["fields"]:
            if(tab["typeName"]=="title"):
                titleNumber += 1
                titlePresence = 1
                name = tab["value"]
                docName = name
                #titles.append({"name":name, "docName": docName})
                titles.append(name)
                
            elif(tab["typeName"]=="dsDescription"):
                descriptionNumber += 1
                descriptionPresence = 1
            
            elif(tab["typeName"]=="producer"):
                publisherNumber += 1
                publisherPresence = 1
                publisherFilledIn.append(docName)
            
            elif(tab["typeName"]=="datasetContact"):
                contactNumber += 1
                contactPresence = 1
                
            elif(tab["typeName"]=="author"):
                creatorNumber += 1
                creatorPresence = 1
                for author in tab["value"]:
                    name = author["authorName"]["value"]
                    if("authorAffiliation" in author):
                        affiliation = author["authorAffiliation"]["value"]
                    else:
                        affiliation = ""
                    authors.append({"name":name, "affiliation":affiliation})      
                
            elif(tab["typeName"]=="keyword"):
                keywordNumber += 1
                keywordPresence = 1
                for keyword in tab["value"]:
                    name = keyword["keywordValue"]["value"]
                    if '*' in name:
                        words = name.split("*")
                        for word in words:
                            word = word.strip()
                            if word != "":
                                keywords.append({"name":word})
                                keywordNumberPerDataset += 1
                                ### mappings
                                if a.get(word)[0][0] >= 0.83:
                                    keywordMatch += 1
                    elif ',' in name:
                        words = name.split(",")
                        for word in words:
                            word = word.strip()
                            if word != "":
                                keywords.append({"name":word})
                                keywordNumberPerDataset += 1
                                ### mappings
                                if a.get(word)[0][0] >= 0.83:
                                    keywordMatch += 1
                    else:
                        keywords.append({"name":name})
                        keywordNumberPerDataset += 1
                        ### mappings
                        if a.get(name)[0][0] >= 0.83:
                            keywordMatch += 1
                               
            elif(tab["typeName"]=="subject"):
                subjectNumber += 1
                subjectPresence = 1
                for subject in tab["value"]:
                    name = subject
                    subjects.append({"name":name})
                    
            elif(tab["typeName"]=="contributor"):
                contributorNumber += 1
                contactPresence = 1
                for contributor in tab["value"]:
                    if("contributorName" in contributor):
                        name = contributor["contributorName"]["value"]
                        contributors.append({"name":name})
                    
            elif(tab["typeName"]=="otherReferences"):
                for CRP in tab["value"]:
                    name = CRP
                    CRPs.append({"name":name})
            
            elif(tab["typeName"]=="kindOfData"):
                typeNumber += 1
                typePresence = 1
                for kindOfData in tab["value"]:
                    name = kindOfData
                    kindsOfData.append({"name":name})
                    
            elif(tab["typeName"]=="dateOfDeposit"):
                dateNumber += 1
                datePresence = 1
                name = tab["value"]
                dates.append({"name":name, "docName": docName})
            
                    
        if("geospatial" in met["datasetVersion"]["metadataBlocks"]):
            for tab in met["datasetVersion"]["metadataBlocks"]["geospatial"]["fields"]:
                geoNumber += 1
                geoPresence = 1
                if(tab["typeName"]=="geographicCoverage"):
                    for geo in tab["value"]:
                        if("otherGeographicCoverage" in geo):
                            coverage = geo["otherGeographicCoverage"]["value"].split(",")
                            for country in coverage:
                                geospatial.append({'name':country.strip()})
                        elif("country") in geo:
                            coverage = geo["country"]["value"].split(",")
                            for country in coverage:
                                geospatial.append({'name':country.strip()})
                            
        for tab in met["datasetVersion"]["files"]:
            contentType = tab["dataFile"]["contentType"]
            fileTypes.append({'name':contentType,"docName": docName})
            
                
        license = met["datasetVersion"]["license"]
        licenses.append({'name':license, "docName": docName})
        if(license != 'NONE'):
            rightPresence = 1
        
        don = {}
        if(titlePresence):
            don["title"] = [{'name':"title", 'value':1}, {'name':"no title", 'value': 0}]
        else:
            don["title"] = [{'name':"title", 'value': 0}, {'name':"no title", 'value': 1}]
        if(creatorPresence):
            don["creator"] = [{'name':"creator", 'value':1}, {'name':"no creator", 'value': 0}]
        else:
            don["creator"] = [{'name':"creator", 'value':0}, {'name':"no creator", 'value': 1}]
        if(subjectPresence):
            don["subjet"] = [{'name':"subject", 'value':1}, {'name':"no subject", 'value': 0}]
        else:
            don["subjet"] = [{'name':"subject", 'value':0}, {'name':"no subject", 'value': 1}]
        if(keywordPresence):
            #don["keyword"] = [{'name':"keyword", 'value':1}, {'name':"no keyword", 'value': 0}]
            don["keyword"] = [{'name':"keyword match", 'value': keywordMatch}, {'name':"no match", 'value': keywordNumberPerDataset-keywordMatch}]
        else:   
            don["keyword"] = [{'name':"keyword", 'value':0}, {'name':"no keyword", 'value': 1}]
        if(descriptionPresence):
            don["description"] = [{'name':"description", 'value':1}, {'name':"no description", 'value': 0}]
        else:
            don["description"] = [{'name':"description", 'value':0}, {'name':"no description", 'value': 1}]
        if(publisherPresence):
            don["publisher"] = [{'name':"publisher", 'value':1}, {'name':"no publisher", 'value': 0 }]
        else:
            don["publisher"] = [{'name':"publisher", 'value':0}, {'name':"no publisher", 'value': 1 }]
        if(contributorPresence):
            don["contributor"] = [{'name':"contributor", 'value':1}, {'name':"no contributor", 'value': 0}]
        else:
            don["contributor"] = [{'name':"contributor", 'value':0}, {'name':"no contributor", 'value': 1}]
        if(datePresence):
            don["date"] = [{'name':"date", 'value':1}, {'name':"no date", 'value': 0}]
        else:
            don["date"] = [{'name':"date", 'value':0}, {'name':"no date", 'value': 1}]
        don["identifier"] = [{'name':"identifier", 'value':1}, {'name':"no identifier", 'value': 0}]
        if(geoPresence):
            don["coverage"] = [{'name':"coverage", 'value':1}, {'name':"no coverage", 'value': 0}]
        else:
            don["coverage"] = [{'name':"coverage", 'value':0}, {'name':"no coverage", 'value': 1}]
        if(rightPresence):
            don["right"] = [{'name':"right", 'value':1}, {'name':"no right", 'value': 0}]
        else:
            don["right"] = [{'name':"right", 'value':0}, {'name':"no right", 'value': 1}]
        if(contactPresence):
            don["contact"] = [{'name':"contact", 'value':1}, {'name':"no right", 'value': 0}]
        else:
            don["contact"] = [{'name':"contact", 'value':0}, {'name':"no right", 'value': 1}]
        if(typePresence):
            don["type"] = [{'name':"type", 'value':1}, {'name':"no type", 'value': 0}]
        else:
            don["type"] = [{'name':"type", 'value':0}, {'name':"no type", 'value': 1}]
        
        donutDataset[datasetID] = don
        
    ##################################@
    idNumber = datasetNumber
    rightNumber = Counter(license['name'] for license in licenses)
    
    #print(CRPs)
    
    #print (licenses)
    #c = Counter(title['name'] for title in titles)
    #c = Counter(subject['name'] for subject in subjects)
    #c = Counter(keyword['name'] for keyword in keywords)
    #c = Counter(author['name'] for author in authors)
    #c = Counter(author['affiliation'] for author in authors)
    #c = Counter(contributor['name'] for contributor in contributors)
    #c = Counter(CRP['name'] for CRP in CRPs)
    c = Counter(geo['name'] for geo in geospatial)
    #c = Counter(fileType['name'] for fileType in fileTypes)
    #c = Counter(fileType['docName'] for fileType in fileTypes)
    #c = Counter(date['name'] for date in dates)
   
    c = Counter(license['name'] for license in licenses)
    #result = [{'name':key, 'value':value} for key,value in c.items()]
    #c = Counter(kindOfData['name'] for kindOfData in kindsOfData)
    result = [{'name':key, 'value':value} for key,value in c.items()]
    
    print(result)
    
    #for the number of document per datasets
    #c1 = Counter(res['value'] for res in result)
    #result1 = [{'name':str(key) + " file", 'value':value} for key,value in c1.items()]
    #print(result1)
    
    #d = {}
    #for k in fileTypes:
    #    d.setdefault(k.get('name'), []).append(k.get('docName'))
    
    #for elem in result:
    #    for e in d:
    #       if(elem['name'] == e):
    #           elem['docName'] = d[e]
   
   
    ###### Donut #####
    donut["title"] = [{'name':"title", 'value':titleNumber}, {'name':"no title", 'value': datasetNumber-titleNumber}]
    donut["creator"] = [{'name':"creator", 'value':creatorNumber}, {'name':"no creator", 'value': datasetNumber-creatorNumber}]
    donut["subjet"] = [{'name':"subject", 'value':subjectNumber}, {'name':"no subject", 'value': datasetNumber-subjectNumber}]
    donut["keyword"] = [{'name':"keyword", 'value':keywordNumber}, {'name':"no keyword", 'value': datasetNumber-keywordNumber}]
    donut["description"] = [{'name':"description", 'value':descriptionNumber}, {'name':"no description", 'value': datasetNumber-descriptionNumber}]
    donut["publisher"] = [{'name':"publisher", 'value':publisherNumber, "dataset": publisherFilledIn}, {'name':"no publisher", 'value': datasetNumber-publisherNumber, "dataset": list(set(titles) - set(publisherFilledIn))}]
    donut["contributor"] = [{'name':"contributor", 'value':contributorNumber}, {'name':"no contributor", 'value': datasetNumber-contributorNumber}]
    donut["date"] = [{'name':"date", 'value':dateNumber}, {'name':"no date", 'value': datasetNumber-dateNumber}]
    donut["identifier"] = [{'name':"identifier", 'value':idNumber}, {'name':"no identifier", 'value': datasetNumber-idNumber}]
    donut["coverage"] = [{'name':"coverage", 'value':geoNumber}, {'name':"no coverage", 'value': datasetNumber-geoNumber}]
    donut["right"] = result
    donut["contact"] = [{'name':"contact", 'value':contactNumber}, {'name':"no right", 'value': datasetNumber-contactNumber}]
    donut["type"] = [{'name':"type", 'value':typeNumber}, {'name':"no type", 'value': datasetNumber-typeNumber}]

    
    print(datasetNumber)
    print(geoNumber)
    print(titleNumber)
    print(creatorNumber)
    #print(creatorAffiliationNumber)
    print(subjectNumber)
    print(keywordNumber)
    print(descriptionNumber)
    print(publisherNumber)
    print(contributorNumber)
    print(dateNumber)
    print(idNumber)
    print(rightNumber)
    print(contactNumber)
    print(typeNumber)
    
    #print(donutDataset)
    
    #print(publisherFilledIn)

# visualization using d3.js
HTML(d3_lib.set_styles(['bar']) + 
'<script src="lib/d3/v3/d3.min.js"></script>' + 
      d3_lib.draw_graph('bar', {'data': result}) )

#save json for d3.js - change the name of the file depending of the variable saved
pathToServer = "path"
with open(pathToServer+'CIAT_dataverseDonutDataset.json', 'w') as outfile:
    json.dump(donutDataset, outfile)



