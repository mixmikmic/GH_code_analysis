import urllib, urllib2
import pprint, json, requests, mysql.connector
from greentranslator.api import GreenTranslator
import mysql.connector

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


flatten = lambda l: [item for sublist in l for item in sublist]

try:
    cnx = mysql.connector.connect(user='tadmin',
                                password='ncats_translator!',
                                database='umls',
                                host='umls.ncats.io')
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)

## Pull in disease to symptom mappings taken from the SI of
## https://www.nature.com/articles/ncomms5212. Takes a bit of time to pull down
DISEASE2SYMPTOMS = [x.split("\t") for x in urllib.urlopen("https://www.nature.com/article-assets/npg/ncomms/2014/140626/ncomms5212/extref/ncomms5212-s4.txt").read().split("\n")]
DISEASE2SYMPTOMS = filter(lambda x: len(x) == 4, DISEASE2SYMPTOMS)

## Define patient classes for easier handling
class Patient:
    def __init__(self, obj):
        self._obj = obj
    def getPatientID(self):
        raise NotImplementedError
    def getSymptoms():
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
        
class PatientFHIR(Patient):
    def __init__(self, obj):
        #pprint.pprint(obj)
        Patient.__init__(self, obj)
        self._symptoms = []
    def getPatientID(self):
        ret = self._obj['subject']['reference']
        return ret.replace("Patient/","")
    def getCodes(self):
        ret = self._obj['code']['coding']
        ret = ['%s#%s' % (x['code'],x['display']) for x in ret]
        return ret
    def getSymptoms(self):
        if len(self._symptoms) == 0:
            try:
                url = "http://ictrweb.johnshopkins.edu/rest/synthetic/Observation?patient="+self.getPatientID()
                response = urllib2.urlopen(url)
                text = response.read()
            except Exceptiom, e:
                return self._symptoms
            obj = json.loads(text)
            if isinstance(obj, dict) and 'symptom' in obj.keys():
                symps = obj['symptom']['coding']
                self._symptoms = [ (x['code'],x['display']) for x in symps ]
        return self._symptoms

    def __str__(self):
        return "%s [%s]" % (self.getPatientID(), '/'.join(self.getCodes()))

## Given disease/condition term, get back ICD codes from OHDSI
def findICD_ohdsi(txt, icd_version = 9):
    if icd_version == 9:
        icd_type = 'ICD9CM'
    elif icd_version == 10:
        icd_type = 'ICD10'
    else: raise Exception("Invalid ICD version specified")    
    url_con = "http://api.ohdsi.org/WebAPI/vocabulary/search"
    headers = {'content-type': 'application/json'}
    params = {"QUERY": txt,
              "VOCABULARY_ID": [icd_type]}
    response = requests.post(url_con, data=json.dumps(params), headers=headers)
    data= json.loads(response.text.decode('utf-8'))
    return [d["CONCEPT_CODE"] for d in data]
print findICD_ohdsi('asthma')

# Get ICD10/ICD9 code for a given string from UMLS. By default we get back ICD10.
def findICD_umls(name, icd_version = 10):
    if icd_version == 9:
        icd_type = 'ICD9CM'
    elif icd_version == 10:
        icd_type = 'ICD10'
    else: raise Exception("Invalid ICD version specified")

    cursor = cnx.cursor()
    query = ("SELECT CUI FROM umls.MRCONSO WHERE STR='"+name+"'")
    cursor.execute(query, ())
    res = "Undef"
    for code in cursor:
        if res=="Undef":
            res = code
    if res != "Undef":
        query = ("SELECT CODE FROM umls.MRCONSO WHERE SAB='"+icd_type+"' AND CUI='"+res[0]+"'")
        cursor.execute(query, ())
        icd10 = "Undef"
        for code in cursor:
            icd10 = code
        return (icd10[0])
    return ("Undef")

print(findICD_umls('Asthma'))
print(findICD_umls('Asthma', 9))

## Given disease name, get back symptoms (defined using MeSH terms) along with TFIDF scores
## Taken from https://www.nature.com/articles/ncomms5212
def disease2symptoms(txt):
    s = filter(lambda x: txt.lower() in x[1].lower(), DISEASE2SYMPTOMS)
    return([(x[0], x[3]) for x in s])
symps = disease2symptoms("Asthma")
print 'Found %s symptom MeSH terms for %s' % (len(symps), "Asthma")

## Get all phenotypes for Asthma from Monarch

def getPhenoTypes(doid):
    url = "https://api.monarchinitiative.org/api/bioentity/disease/DOID%3A"+doid+"/phenotypes/?rows=20&fetch_objects=false&unselect_evidence=true"
    response = requests.get(url)
    #print response.text.decode('utf-8')
    res = json.loads(response.text.decode('utf-8'))
    phenotypes = []
    for o in res['objects']:
        cursor = cnx.cursor()
        query = ("SELECT CUI,STR FROM umls.MRCONSO WHERE CODE='"+o+"' AND SAB='HPO'")
        cursor.execute(query, ())
        cui_str = ("Undef","Undef")
        for code in cursor:
            cui_str = code
        icd = findICD_ohdsi(cui_str[1], 10)
        #print cui_str
        #query = ("SELECT CODE FROM umls.MRCONSO WHERE CUI='"+cui_str[0]+"' AND (SAB='ICD10' OR SAB='ICD10CM')")
        #print query
        #cursor.execute(query, ())
        #res_code = ("Undef","Undef")
        #for code in cursor:
        #    res_code = code
        #print res_code
        phenotypes.append((cui_str[1],icd))
    return phenotypes

asthmaPhenotypes = getPhenoTypes("7148")
asthmaPhenotypeMap = {}
for x in asthmaPhenotypes:
    if x[1] != 'Undef':
        asthmaPhenotypeMap[x[0]] = x[1]

#pprint.pprint(asthmaPhenotypes)

print 'Mapped phenotype terms for %s found in Monarch to %s unique ICD9 codes' % ("Asthma", len(asthmaPhenotypes))

## Functions to retreive patients from different sources - currently FHIR & UNC
def findPatients_fhir(code, count=1000):
    try:
        response = urllib2.urlopen("http://ictrweb.johnshopkins.edu/rest/synthetic/Condition?icd_10="+code+"&_count=%d" % (count))
        text = response.read()
    except Exception, e:
        raise Exception(e)
    objs = json.loads(text)
    if len(objs) > 0:
        patients = [PatientFHIR(x) for x in objs['entry']]
    else: patients = []
    return patients

def findPatients_unc(age='8', sex='male', race='white', location='OUTPATIENT'):
    query = GreenTranslator ().get_query()
    return query.clinical_get_patients(age, sex, race, location)            

asthmaCodes = findICD_umls("asthma") # We go with ICD10 codes
## get patients with asthma. First from FHIR, then with UNC
p_fhir = flatten(filter(lambda x: len(x) > 0, [findPatients_fhir(icd) for icd in asthmaCodes]))
##p_unc = findPatients_unc() # TODO needs to be updated to latest code

asthmaSymptoms = disease2symptoms("asthma")
print 'Found %s symptom MeSH terms for %s' % (len(symps), "asthma")
asthmaSymptomCodes = filter(lambda x: x != 'U', [findICD_umls(x[0], 10) for x in symps])


tmp2 = flatten([findICD_ohdsi(x[0], 10) for x in symps])
asthmaSymptomCodes.extend(tmp2)

asthmaSymptomCodes = list(set(asthmaSymptomCodes))
print 'Mapped to %d unique ICD10 codes' % (len(asthmaSymptomCodes))

#print asthmaSymptomCodes

def mapSymptom(x):
    try:
        r = x + "("+asthmaPhenotypeMap(x)+")"
        print "Here"
    except:
        r = x
    return r

#for x in p_fhir:
#    print "Patient"
#    print x
#    print "Symptoms"
#    print x.getSymptoms()

asthmaPatients = [(x.getPatientID(), [y[1] for y in x.getSymptoms()]) for x in p_fhir]
#print asthmaPatients

forBroad = {key:value for (key, value) in asthmaPatients}
#print forBroad
jsonForClustering = json.dumps(forBroad)
o = open('json-for-clustering.json', 'w')
o.write(jsonForClustering)
o.close()

universe = list(set(flatten([x[1] for x in asthmaPatients])))
universe.sort()
data = []
for pid,symptoms in asthmaPatients:
    vec = [0 for x in range(0,len(universe)+1)]
    vec[0] = pid
    for i in range(0, len(universe)):
        if universe[i] in symptoms:
            vec[i+1] = 1
    data.append(vec)

header = ['PID']
header.extend(universe)
data = pd.DataFrame(data,columns=header).set_index('PID')

import seaborn as sns
get_ipython().magic('matplotlib inline')
cg = sns.clustermap(data, metric='hamming', z_score=None)
for text in cg.ax_heatmap.get_yticklabels():
    text.set_rotation('horizontal')
cg    

