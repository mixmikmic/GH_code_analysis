import pandas as pd #Pandas helps transform our data into data frames
import matplotlib.pyplot as plt #Needed for plotting
import operator #Used in sorting lists
import numpy as np #Numpy helps with processing arrays

#The following packages from sci-kit learn will help with building our model and evaluating it:
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#Directory with all relevant files
currDir = 'C:/Users/Arielle/OneDrive/CompBioProjects/PredictADRs/InputFiles/'

#File from SIDER with ADRs
ADRfile = open(currDir + 'meddra_all_se.tsv', 'r')

#File that contains conversion from Pubchem ID to Drugbank ID (taken from: https://raw.githubusercontent.com/dhimmel/drugbank/e8567eed2dd48ae0694a0960c518763a777845ff/data/mapping/pubchem.tsv)
conversionFile = open(currDir + 'pubchem.tsv', 'r')

#Dictionary with keys representing Pubchem IDs, and values representing the corresponding Drugbank IDs.
PubchemToDB = {}

for line in conversionFile:
    line = line.rstrip()
    cols = line.split('\t')
    db_id = cols[0]
    pubchem_id = cols[1]
    if pubchem_id not in PubchemToDB:
        #Some Pubchem IDs may appear more than once, so we will just consider the first one that appears in the file
        PubchemToDB[pubchem_id] = db_id
        

#Dictionary with keys representing Drugbank IDs, and values representing a dictionary of ADRs.
DBtoADR = {}

#Dictionary with keys represting ADRs, and values representing the frequency of the ADRs.
ADRtoFreq = {}

for line in ADRfile:
    line = line.rstrip('\n')
    cols = line.split('\t')
    CID = cols[0]
    term = cols[3]
    if term == "PT": #We are using the "preferred term" of the ADR
        CID_num_full = CID[3:]
        pubchem_id = int(CID_num_full) - 100000000
        pubchem_id = str(pubchem_id)
        if pubchem_id in PubchemToDB: #We only want the ADR info for drugs that have a Drugbank ID
            DB = PubchemToDB[pubchem_id]
            adr = cols[5]
            if DB in DBtoADR: #Record the ADRs associated with each drug
                adr_dict = DBtoADR[DB]
                adr_dict[adr] = "NA"
                DBtoADR[DB] = adr_dict
            else:
                DBtoADR[DB] = {}
                DBtoADR[DB][adr] = "NA"
            if adr in ADRtoFreq: #Record the ADR frequencies
                currFreq = ADRtoFreq[adr]
                ADRtoFreq[adr] = currFreq + 1
            else:
                ADRtoFreq[adr] = 1

#Convert ADRtoFreq dictionary to a data frame
ADRtoFreq_list = [] 
ADRtoFreq_list.append(ADRtoFreq)
ADRtoFreq_df = pd.DataFrame(ADRtoFreq_list)

ADRtoFreq_df.head()

#Combined target, enzyme, carrier, and transporter gene file from DrugBank. This file contain the genes associated with each drug.
drugFile = open(currDir + 'drug_allTypes.csv', 'r')

#File to convert the GO term number to its actual name (the GO terms are represented in their numerical form in the input files)
GOtermIDfile = open(currDir + 'GO.terms_alt_ids', 'r')

#Files containing the GO terms associated with all genes in a species
humanFile = currDir + 'goa_human.gaf'
ecoliFile = currDir + 'gene_association.ecocyc'
yeastFile = currDir + 'gene_association.yeast'
leishmaniaFile = currDir + 'gene_association.GeneDB_Lmajor'
plasmodiumFile = currDir + 'gene_association.GeneDB_Pfalciparum'
trypanosomaFile = currDir + 'gene_association.GeneDB_Tbrucei'
agrobacteriumFile = currDir + 'gene_association.PAMGO_Atumefaciens'
aspergillusFile = currDir + 'gene_association.aspgd'
pseudomonasFile = currDir + 'gene_association.pseudocap'
celegansFile = currDir + 'gene_association.wb'
mouseFile = currDir + 'gene_association.mgi'
cowFile = currDir + 'goa_cow.gaf'
ratFile = currDir + 'gene_association.rgd'

#Dictionary with keys representing species, and values representing GO term file associated with that species
speciesList = {}
speciesList['Human'] = humanFile
speciesList['sapiens'] = humanFile
speciesList['Escherichia'] = ecoliFile
speciesList['Yeast'] = yeastFile
speciesList["Baker's yeast"] = yeastFile
speciesList['Leishmania'] = leishmaniaFile
speciesList['Plasmodium'] = plasmodiumFile
speciesList['Trypanosoma'] = trypanosomaFile
speciesList['Agrobacterium'] = agrobacteriumFile
speciesList['Aspergillus'] = aspergillusFile
speciesList['Pseudomonas'] = pseudomonasFile
speciesList['elegans'] = celegansFile
speciesList['Mouse'] = mouseFile
speciesList['taurus'] = cowFile
speciesList['Rat'] = ratFile

#Dictionary with keys representing Drugbank IDs, and values representing a dictionary of GO terms associated with those drugs.
DBtoGOterms = {}

#Dictionary with keys represting GOterms, and values representing the frequency of those GO terms.
GOtermToFreq = {}

#Species to Gene Name to GO terms (all dictionaries).  This is needed in ordered to obtain the GOterms for each Drugbank ID.
speciesGeneGOterms = {}

#Dictionary with keys representing GO terms, and values representing the actual labels of those GO terms.
GOtermToID = {}

#Obtain the name/identification for each GO term
for line in GOtermIDfile:
    line = line.rstrip('\n')
    cols = line.split('\t')
    firstChar = line[0]
    if firstChar != "!":
        primTerm = cols[0]
        secTerm = cols[1]
        ID = cols[2]
        GOtermToID[primTerm] = ID

#Get all GOterms associated with all genes from a species. Returns a gene -> GO terms dictionary. 
def ParseSpeciesFile(speciesName):
    speciesFile = speciesList[speciesName]
    f = open(speciesFile, 'r')
    geneToGOterms = {}
    for line in f:
        line = line.rstrip('\n')
        cols = line.split('\t')
        if cols[0][0] != '!':
            gene = cols[2]
            GOterm = cols[4]
            if GOterm in GOtermToID: #Make sure GO term has an identifiable name
                ID = GOtermToID[GOterm]
                possibleNot = cols[3]
                if possibleNot != 'NOT':
                    GOdict = {} #Record the GO term names into this 'GOdict' dictionary
                    if gene in geneToGOterms: #See if the GO terms for this gene have already been recorded
                        GOdict = geneToGOterms[gene]
                    GOdict[ID] = 'NA' #Add new GO term
                    geneToGOterms[gene] = GOdict #Store updated GO term dictionary into the "geneToGOterms" dictionary
    return geneToGOterms

#Get the GO terms associated with a specific gene from a particular species.  Returns a dictionary of GO terms.
def GetGOterms(speciesName, gene):
    geneToGOterms = speciesGeneGOterms[speciesName]
    currGOterms = {}
    for posGene in geneToGOterms:
        if gene == posGene:
            currGOterms = geneToGOterms[gene]
            break
    return currGOterms

#For each species, obtain all genes and associated GO terms
for species in speciesList:
    speciesGeneGOterms[species] = ParseSpeciesFile(species)

#Get the GO terms associated with each drug
for line in drugFile:
    line = line.rstrip('\n')
    cols = line.split(',')
    if cols[0] != 'ID':
        currGene = cols[2] #A target, enzyme, carrier, or transporter
        species = cols[11] #Species associated with the gene
        drugs = cols[12] #Drugs associated with the gene, separated by semi-colons
        #Determine actual species (nomenclature can be inconsistent in this file)
        checkSpecies = ''
        speciesSplit = species.split(' ')
        for term in speciesSplit:
            if term in speciesList:
                checkSpecies = term
        currSpecies = ''
        if species in speciesList:
            currSpecies = species
        elif checkSpecies != '':
            currSpecies = checkSpecies
        if currSpecies != '': #Only proceed if we have properly identified the species
            GOtermDict = GetGOterms(currSpecies, currGene) #Get the GO terms associated with this species and gene
            drugsSplit = drugs.split(';') #Turn drug info into a list
            for d in drugsSplit:
                d = d.strip()
                #Record the GO terms associated with these drugs
                if d in DBtoGOterms: #Check if drug is already in DBtoGOterms dictionary
                    currDict = DBtoGOterms[d]
                    for GOterm in GOtermDict: #Add new GO terms
                        currDict[GOterm] = 'NA'
                    DBtoGOterms[d] = currDict
                else:
                    DBtoGOterms[d] = GOtermDict
                #Record frequency of GO terms
                for GOterm in GOtermDict:
                    if GOterm in GOtermToFreq:
                        currFreq = GOtermToFreq[GOterm]
                        GOtermToFreq[GOterm] = currFreq + 1
                    else:
                        GOtermToFreq[GOterm] = 1

#Examine frequency of GO terms
#Convert GOtermToFreq dictionary to a data frame
GOtermToFreq_list = [] 
GOtermToFreq_list.append(GOtermToFreq)
GOtermToFreq_df = pd.DataFrame(GOtermToFreq_list)

GOtermToFreq_df.head()

#List containing drugs with ADR info and with GO term features.
overlappedDrugs = []

#List containing GO term features of drugs that also have ADR info. 
relevantFeatures = []

#Obtain list of drugs and features to use in the model
for drug in DBtoADR:
    if drug in DBtoGOterms:
        overlappedDrugs.append(drug)
        features = DBtoGOterms[drug]
        for GOterm in features:
            if GOterm not in relevantFeatures:
                relevantFeatures.append(GOterm)
                
print('Number of drugs: ' + str(len(overlappedDrugs)))
print('Number of features: ' + str(len(relevantFeatures)))
                

tremorFreq = ADRtoFreq['Tremor']
numDrugs = len(overlappedDrugs)

tremorFraction = int(tremorFreq)/int(numDrugs)

print('Fraction of drugs with a tremor ADR: %0.2f'% tremorFraction)

#The ADR we are trying to predict
se = "Tremor"

#List of labels - which drugs have the ADR, and which don't?
y_list = []

#List of list containing feature information for each drug.  This will then be converted to a data frame.
x_listOflist = []

for drug in overlappedDrugs:
    ADRs = DBtoADR[drug] #Get dictionary of ADRs associated with this drug that we generated earlier
    se_status = 0 #First, assume drug does not have this ADR
    if se in ADRs: #If drug is in ADR, then we'll record this as a "1"
        se_status = 1 
    y_list.append(se_status) #Add the "status" of this ADR to the label list
    features = DBtoGOterms[drug] #Get dictionary of features associated with this drug that we generated earlier
    featureList = []
    for feature in relevantFeatures: #For each feature, determine whether the drug in question is associated with that feature.
        if feature in features:
            featureList.append(1)
        else:
            featureList.append(0)
    x_listOflist.append(featureList) #Append this list to our main feature info list

#Convert list of list to data frame
x_df = pd.DataFrame(x_listOflist)

#Split data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(x_df, y_list, test_size=0.2, random_state=1)

#Enter our Random Forest Classifier into a pipeline; this facilitates cross-validation.
pipeline = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=2017))

#Declare hyperparameters that we want to adjust
hyperparameters = {'randomforestclassifier__max_features' : ['auto', 'log2'],
                  'randomforestclassifier__max_depth': [None, 1, 3, 5],
                    'randomforestclassifier__min_samples_leaf': [1, 5, 10]}
 
#Use cross-validation pipeline to tune hyperparameters
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

#Fit our model
clf.fit(X_train, y_train)

#Determine how accurate the model is on the test data set
score = clf.score(X_test, y_test)

print('Overall accuracy: ' + str(score) + '\n')

#Predictions of the testing data
pred = clf.predict(X_test)

a = confusion_matrix(y_test, pred)
print('Confusion matrix:')
print(a)
print('\n')

print('Additional metrics:')
print(metrics.classification_report(y_test, pred))

#Predict probabilities of positive class
y_score = clf.predict_proba(X_test)[:,1]

#Calculate false positive rate and true positive rate at different probability thresholds
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score)

#Calculate the AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

#Plot the ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.tick_params(
    axis='x',          
    top='off')        
plt.tick_params(
    axis='y',          
    right='off')      
plt.show()

FeatureImportanceList = []

#Extract feature importances
fi = list(zip(X_train, clf.best_estimator_.named_steps['randomforestclassifier'].feature_importances_))
for f in fi:
    index = f[0]
    imp = f[1]
    GOterm = relevantFeatures[int(index)]
    GOterm_imp = (GOterm, imp)
    FeatureImportanceList.append(GOterm_imp)

#Sort features based on score
sortedFeatureImportanceList = sorted(FeatureImportanceList, key=operator.itemgetter(1), reverse = True)

#Examine the top 10 feature importances
GOterms = list(zip(*sortedFeatureImportanceList[:10]))[0]
scores = list(zip(*sortedFeatureImportanceList[:10]))[1]

#Make axes labels evenly space
ind = np.arange(len(GOterms))

#Generate barplot of top 10 feature importances
fig, ax = plt.subplots()
ax.barh(ind,scores,0.5, align='center')
ax.set_yticks(ind)
ax.set_yticklabels(GOterms)
ax.invert_yaxis()
plt.title('Feature Importances')
plt.xlabel('Score')
plt.tick_params(
    axis='x',          
    top='off',
    bottom='off')         
plt.tick_params(
    axis='y',        
    left='off',   
    right='off')        
fig.set_facecolor('white')
plt.autoscale()
plt.show()

