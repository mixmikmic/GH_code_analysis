# This notebook takes lists of lobbyist registration termination dates from the City of Austin's data portal and uses
# them to calculate when the lobbyists were hired and how long they remained registered to represent the same clients.

import csv, copy
import datetime

# These are from the Datamade dedupe example

from future.builtins import next

import os
import re
import logging
import optparse

import dedupe
from unidecode import unidecode

def addLobby(file, lobbyistList):
    newFile = open(file)
    newReader = csv.reader(newFile)
    newData = list(newReader)
    for row in newData[1:]:
        endDate = datetime.datetime.strptime(row[3], "%m/%d/%Y").date()
        startDate = endDate - datetime.timedelta(days=365)
        fullName = row[1] + ' ' + row[0] # merging separate first and last name fields
        lobbyistList.append([startDate, endDate, fullName, row[4], row[6], row[5]])

lobbyists = []
files = ['./data/lobbyists2014.csv', './data/lobbyists2015.csv', './data/lobbyists2015-10.csv', './data/lobbyists2016.csv', './data/lobbyists2016-04.csv']

for file in files:
    addLobby(file, lobbyists)

outputFile = open('./data/lobbyistFromCity.csv', 'w', newline='')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['Start', 'End', 'Lobbyist', 'Client', 'Industry', 'ClientAddress'])
for row in lobbyists:
    outputWriter.writerow(row)
outputFile.close()

# copied from http://datamade.github.io/dedupe-examples/docs/csv_example.html

input_file = './data/lobbyistFromCity.csv'
output_file = './data/lobbyistDedupe.csv'
settings_file = './data/csv_example_learned_settings' # delete this from the directory to train some more
training_file = './data/trainingClient.json' # delete this only if you want to start over with training
# in the example there was a new output file defined here.


def preProcess(column):

    try : # python 2/3 string differences
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()

# If data is missing, indicate that by setting the value to None

    if not column:
        column = None
    return column



def readData(filename):

# Read in our data from a CSV file and create a dictionary of records, 
# where the key is a unique record ID and each value is dict

    data_d = {}
    i = 0
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = i
            i = i + 1
            data_d[row_id] = dict(clean_row)

    return data_d

print('importing data ...')
data_d = readData(input_file)



# If a settings file already exists, we'll just load that and skip training
if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f)
else:
    # ## Training

    # Define the fields dedupe will pay attention to
    fields = [
        {'field' : 'Client', 'type': 'String'},
        {'field' : 'Industry', 'type': 'String'},
        {'field' : 'ClientAddress', 'type': 'String'},
        ]

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    # To train dedupe, we feed it a sample of records.
    deduper.sample(data_d, 15000)

    labeled_examples = {'match': [
            ({"Client": "met center nyctex phase ii, ltd.", 
                    "Industry": "property owner", 
                    "ClientAddress": "611 west 15th street austin, texas 78701 (30.278703917000485, -97.74603822499967)"},
                    {"Client": "met center ii partners", 
                    "Industry": "zoning", 
                    "ClientAddress": "1135 west 6th street, suite 120 austin, texas 78703 (30.272632, -97.756889)"})
                ],
                    "distinct": [
                    ({
                    "Client": "robert ehrlich", 
                    "Industry": "property owner", 
                    "ClientAddress": "601 w. 38th street, suite 206 austin, texas 78731 (30.30888504200044, -97.75021326899969)"},
                    {"Client": "robert ross", 
                    "Industry": "property owner", 
                    "ClientAddress": "1601 west 38th street, suite 108 austin, texas 78731 (30.30888504200044, -97.75021326899969)"}),
                
                    ({"Client": "robert rock", 
                    "Industry": "property owner", 
                    "ClientAddress": "5011 burnet road austin, texas 78759 (30.32190613800003, -97.73929964899997)"},
                    {"Client": "robert ross", 
                    "Industry": "property owner", 
                    "ClientAddress": "1601 west 38th street, suite 108 austin, texas 78731 (30.30888504200044, -97.75021326899969)"}),
           
                    ({"Client": "fagan, dennis, still & moving pictures", 
                    "Industry": "property owner", 
                    "ClientAddress": "1601 west 38th street, suite 201 austin, texas 78731 (30.308718, -97.749802)"},
                    {"Client": "wiman, sophie", 
                    "Industry": "property owner", 
                    "ClientAddress": "1601 west 38th street, suite 12 austin, texas 78731 (30.308749, -97.749866)"})            
                ]}
                   
    deduper.markPairs(labeled_examples)
    
    # If we have training data saved from a previous run of dedupe,
    # look for it and load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.readTraining(f)

    # ## Active learning
    # Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.
    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print('starting active labeling...')

    dedupe.consoleLabel(deduper)

    # Using the examples we just labeled, train the deduper and learn
    # blocking predicates
    deduper.train()

    # When finished, save our training to disk
    with open(training_file, 'w') as tf:
        deduper.writeTraining(tf)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_file, 'wb') as sf:
        deduper.writeSettings(sf)
        
# Find the threshold that will maximize a weighted average of our
# precision and recall.  When we set the recall weight to 2, we are
# saying we care twice as much about recall as we do precision.
#
# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold = deduper.threshold(data_d, recall_weight=1)


# ## Clustering

# `match` will return sets of record IDs that dedupe
# believes are all referring to the same entity.

print('clustering...')
clustered_dupes = deduper.match(data_d, threshold)

print('# duplicate sets', len(clustered_dupes))

# ## Writing Results

# Write our original data back out to a CSV with a new column called 
# 'Cluster ID' which indicates which records refer to each other.

cluster_membership = {}
cluster_id = 0
for (cluster_id, cluster) in enumerate(clustered_dupes):
    id_set, scores = cluster
    cluster_d = [data_d[c] for c in id_set]
    canonical_rep = dedupe.canonicalize(cluster_d)
    for record_id, score in zip(id_set, scores):
        cluster_membership[record_id] = {
            "cluster id" : cluster_id,
            "canonical representation" : canonical_rep,
            "confidence": score
        }

singleton_id = cluster_id + 1

with open(output_file, 'w') as f_output, open(input_file) as f_input:
    writer = csv.writer(f_output)
    reader = csv.reader(f_input)

    heading_row = next(reader)
    heading_row.insert(0, 'confidence_score')
    heading_row.insert(0, 'Cluster ID')
    canonical_keys = canonical_rep.keys()
    for key in canonical_keys:
        heading_row.append('canonical_' + key)

    writer.writerow(heading_row)

    i = 0
    for row in reader:
        row_id = i
        i = i + 1
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]["cluster id"]
            canonical_rep = cluster_membership[row_id]["canonical representation"]
            row.insert(0, cluster_membership[row_id]['confidence'])
            row.insert(0, cluster_id)
            for key in canonical_keys:
                row.append(canonical_rep[key].encode('utf8'))
        else:
            row.insert(0, None)
            row.insert(0, singleton_id)
            singleton_id += 1
            for key in canonical_keys:
                row.append(None)
        writer.writerow(row)


def addDLobby(file, lobbyistList):
    newFile = open(file)
    newReader = csv.reader(newFile)
    newData = list(newReader)
    # Don't dump the address field because we're going to reuse it later 
    # to link lobbyist client records to campaign donation records.
    for row in newData[1:]:
        endDate = datetime.datetime.strptime(row[3], "%Y-%m-%d").date()
        startDate = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
        lobbyistList.append([startDate, endDate, row[4], row[5], row[6], int(row[0]), row[7], row[1]])

lobbyists = []
files = ['./data/lobbyistDedupe.csv']

for file in files:
    addDLobby(file, lobbyists)

q = type(lobbyists[1][5])
print(q)

# Cleaning client names. Many of these lines are probably obsolete now because I added them before I made the
# dedupe library part of the process.
    
for lobbyist in lobbyists:
    lobbyist[3] = lobbyist[3].rstrip()
    # get rid of ' c\o' followed by anything to the end of the line
    lobbyist[3] = re.sub(',? (c/o|attn:|Attn:).*$', '', lobbyist[3])
    lobbyist[3] = re.sub('\,? L\.?L?\.?(C|P)\.?\,?$', '', lobbyist[3])
    lobbyist[3] = re.sub(',? I(n|m|N)(c|C)(\.|(orporated))? ?$', '', lobbyist[3])
    lobbyist[3] = re.sub(',? L(t|T)(d|D).?$', '', lobbyist[3])
    lobbyist[3] = re.sub('\, P\.?C\.?,?$', '', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace("Acheive","Achieve")
    lobbyist[3] = lobbyist[3].replace('  ',' ')
    lobbyist[3] = lobbyist[3].replace(' - Austin','')
    lobbyist[3] = lobbyist[3].replace(' (Vance Elliott)','')
    lobbyist[3] = lobbyist[3].replace('2208 Lake Austin, LLC','2208 Lake Austin')
    lobbyist[3] = lobbyist[3].replace('Amini, Ashley','Ashley Amini')
    lobbyist[3] = lobbyist[3].replace('A T & T','AT&T')
    lobbyist[3] = lobbyist[3].replace('AT & T','AT&T')
    lobbyist[3] = lobbyist[3].replace('A T and T','AT&T')
    lobbyist[3] = lobbyist[3].replace('AT and T','AT&T')
    lobbyist[3] = re.sub('AT&T.*$','AT&T', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('AT& T','AT&T')
    lobbyist[3] = lobbyist[3].replace(" & ", " and ")
    lobbyist[3] = lobbyist[3].replace('Attal, Deborah and Gary','Deborah and Gary Attal')
    lobbyist[3] = lobbyist[3].replace('Austin Elm Terrace LP (Steve Beuerlein)','Austin Elm Terrace LP')
    lobbyist[3] = lobbyist[3].replace('Austin Elm Terrace, LP Burlington Ventures','Austin Elm Terrace LP')
    lobbyist[3] = lobbyist[3].replace('Autoreturn','AutoReturn')
    lobbyist[3] = lobbyist[3].replace('Behringer Harvard Terrace','Behringer Harvard')
    lobbyist[3] = lobbyist[3].replace('Blatt, Jeff','Jeff Blatt')
    lobbyist[3] = re.sub('David Booth$','David and Suzanne Booth',lobbyist[3])
    lobbyist[3] = re.sub('Booth, David and Suzanne$','David and Suzanne Booth',lobbyist[3])
    lobbyist[3] = re.sub('Booth, David$','David and Suzanne Booth',lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Booth, Suzanne Deal','David and Suzanne Booth')
    lobbyist[3] = lobbyist[3].replace('Booth and Suzanne','Booth')
    lobbyist[3] = lobbyist[3].replace('Bridges, Will','Will Bridges')
    lobbyist[3] = lobbyist[3].replace('Burris, Gene','Gene Burris')
    lobbyist[3] = lobbyist[3].replace('Browning, Karen','Karen Browning')
    lobbyist[3] = lobbyist[3].replace('Follett, Brian','Brian Follett')
    lobbyist[3] = lobbyist[3].replace('Byrne, Dan','Dan Byrne')
    lobbyist[3] = lobbyist[3].replace('Calderon, Alex and Mark','Alex and Mark Calderon')
    lobbyist[3] = lobbyist[3].replace('Alex Calderon Mark Calderon','Alex and Mark Calderon')
    lobbyist[3] = lobbyist[3].replace('Caledona Properties','Caledonia Properties')
    lobbyist[3] = lobbyist[3].replace('Capital Metropolitan Transportation Authority','Capital Metro')
    lobbyist[3] = lobbyist[3].replace('Cathcart, Mark','Mark Cathcart')
    lobbyist[3] = lobbyist[3].replace('Cavanaugh, Roy','Roy Cavanaugh')
    lobbyist[3] = lobbyist[3].replace('Castle Hill Partners','Castle Hill Management')
    lobbyist[3] = re.sub('Carollo$','Carollo Engineers', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Fisher, Chereen','Chereen Fisher')
    lobbyist[3] = lobbyist[3].replace('Dildy, Charles S. and Bertha Magdelan Steger, Trustees','Charles S. Dildy and Bertha Magdelan Steger Dildy, Trustees')
    lobbyist[3] = lobbyist[3].replace('Dorrance, Charles','Charles Dorrance')
    lobbyist[3] = re.sub('Colina West.*$','Colina West Real Estate', lobbyist[3])
    lobbyist[3] = re.sub('Lennar.*$','Lennar Homes', lobbyist[3])
    lobbyist[3] = re.sub('Amstar.*$','Amstar Group', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Crown Castle USA', 'Crown Castle')
    lobbyist[3] = re.sub(' Corporation$','', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Cypress Real Estate Advisors','Cypress Realty')
    lobbyist[3] = lobbyist[3].replace('Cypress VI Reit','Cypress Realty')
    lobbyist[3] = lobbyist[3].replace('Davidson, David','David Davidson')
    lobbyist[3] = lobbyist[3].replace('DeRoeck, Walter A.','Walter A. DeRoeck')
    lobbyist[3] = lobbyist[3].replace('Development Company','Development')
    lobbyist[3] = lobbyist[3].replace('English, Toria and Blake','Toria and Blake English')
    lobbyist[3] = re.sub('Eureka$','Eureka Holdings', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('George, Anthony','Anthony George')
    lobbyist[3] = re.sub('^HEB$','H-E-B', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Hardeman, Bryan','Bryan Hardeman')
    lobbyist[3] = lobbyist[3].replace('HEB Grocery Stores','H-E-B')
    lobbyist[3] = lobbyist[3].replace('H-E-B Grocery Stores','H-E-B')
    lobbyist[3] = lobbyist[3].replace('Gerbracht Heidi','Heidi Gerbracht') 
    lobbyist[3] = lobbyist[3].replace('Highland Resources','Highland Management')
    lobbyist[3] = lobbyist[3].replace('Jackson, Glenn','Glenn Jackson')
    lobbyist[3] = lobbyist[3].replace(', Jr.',' Jr.')
    lobbyist[3] = lobbyist[3].replace('JP Morgan Chase Bank - Mail Code IL1-0930, ATTN Retail Portfolio Manager','JP Morgan Chase Bank')
    lobbyist[3] = lobbyist[3].replace('Gorence, Kenneth','Kenneth Gorence')
    lobbyist[3] = lobbyist[3].replace('Lambert, Liz','Liz Lambert')
    lobbyist[3] = lobbyist[3].replace('Laney, Terry','Terry Laney')
    lobbyist[3] = lobbyist[3].replace('Moody, Linda', 'Linda Moody') 
    lobbyist[3] = lobbyist[3].replace('Lindy Moody', 'Linda Moody')
    lobbyist[3] = lobbyist[3].replace('Multiplyer', 'Multiplier')
    lobbyist[3] = lobbyist[3].replace('Lopez, Edward','Edward Lopez')
    lobbyist[3] = re.sub('Met Center.*$', 'Met Center', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Momin, Sohil','Sohil Momin')
    lobbyist[3] = lobbyist[3].replace('Moritz,Jim','Moritz, Jim')
    lobbyist[3] = re.sub('Properties #\d', 'Properties', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Patel, Vijay','Vijay Patel')
    lobbyist[3] = lobbyist[3].replace('Reality Investor','Realty Investor')
    lobbyist[3] = lobbyist[3].replace('Reddehase, Eric','Eric Reddehase')
    if lobbyist[3] == 'Redflex Guardian': lobbyists.remove(lobbyist)
    if lobbyist[3] == 'RedLeaf Highland': lobbyists.remove(lobbyist)
    lobbyist[3] = lobbyist[3].replace('Reynolds, Cary and Cynthia','Reynolds, Cary and Cynthia')
    lobbyist[3] = lobbyist[3].replace('Robert P.Wills','Robert P. Wills')
    lobbyist[3] = lobbyist[3].replace('Wills, Robert P.','Robert P. Wills')
    lobbyist[3] = lobbyist[3].replace('Scarbrough Wilson, Margaret','Margaret Scarbrough Wilson')
    lobbyist[3] = lobbyist[3].replace('Wilson, Randy','Randy Wilson')
    lobbyist[3] = lobbyist[3].replace('Sayers, Scott','Scott Sayers')
    lobbyist[3] = lobbyist[3].replace('Schmidt, Robert','Robert Schmidt')
    lobbyist[3] = re.sub('Seton$','Seton Healthcare', lobbyist[3])
    lobbyist[3] = re.sub('Seton Healthcare.*$','Seton Healthcare', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Slover, Christopher','Christopher Slover')
    lobbyist[3] = lobbyist[3].replace('Smith, Ford Jr.','Ford Smith Jr.')
    lobbyist[3] = lobbyist[3].replace('Schoenbaum, James','James Schoenbaum')
    lobbyist[3] = lobbyist[3].replace('Greenberg, Steve','Steve Greenberg')
    lobbyist[3] = re.sub('Partners (I|II|III|IV|V|VI|VII|VIII) ', 'Partners ', lobbyist[3])
    lobbyist[3] = re.sub('Realty (I|II|III|IV|V|VI|VII|VIII) ', 'Realty ', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace(' Limited Partnership','')
    lobbyist[3] = lobbyist[3].replace(' PHASE II','')
    lobbyist[3] = lobbyist[3].replace('Reynolds, Cary and Cynthia','Cary and Cynthia Reynolds')
    lobbyist[3] = lobbyist[3].replace('Samsung Austin Semiconductor','Samsung')
    lobbyist[3] = lobbyist[3].replace("St. David's Community Health Foundation Initiatives", "St. David's Foundation")
    lobbyist[3] = lobbyist[3].replace('Schneider, James','James Schneider')
    lobbyist[3] = lobbyist[3].replace('Simmons Vedder and Co.','Simmons Vedder Partners')
    lobbyist[3] = lobbyist[3].replace('Simon Properties', 'Simon Property Group')
    lobbyist[3] = lobbyist[3].replace('Spire Realty Group','Spire Realty')
    lobbyist[3] = lobbyist[3].replace('SXSW Properties','SXSW')
    lobbyist[3] = lobbyist[3].replace('Saleem Tawill','Saleem Tawill')
    lobbyist[3] = re.sub(' Corp$', '', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace("St. David's Healthcare Center", "St. David's Foundation")
    lobbyist[3] = re.sub('Stratford$', 'Stratford Land', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Snyder, Suzanne','Suzanne Snyder')
    lobbyist[3] = re.sub('Lone Star Rail$', 'Lone Star Rail', lobbyist[3])
    lobbyist[3] = re.sub('Tantallon Austin$','Tantallon Austin Hotel', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Telvending','Televending')
    lobbyist[3] = lobbyist[3].replace('Texs Aggregates and Concrete Association','Texas Aggregates and Concrete Association')
    lobbyist[3] = lobbyist[3].replace('Theriot, Robert H.','Robert H. Theriot')
    lobbyist[3] = lobbyist[3].replace('Trinity Insurance Services','Trinity Insurance Group')
    lobbyist[3] = lobbyist[3].replace('Turner,Ben','Turner, Ben')
    lobbyist[3] = lobbyist[3].replace('Ben, Consort','Ben')
    lobbyist[3] = lobbyist[3].replace('Turner, Rob','Robert Turner')
    lobbyist[3] = lobbyist[3].replace('Naddef, Wilfred','Wilfred Naddef')
    lobbyist[3] = lobbyist[3].replace('The County Line','County Line')
    lobbyist[3] = lobbyist[3].replace('University of Texas, Board of Regents', 'The University of Texas System')
    lobbyist[3] = re.sub('URSTCO$','URSTCO GP', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace(' (USA)','')
    lobbyist[3] = lobbyist[3].replace('Vega, Shirley','Shirley Vega')
    lobbyist[3] = lobbyist[3].replace('Vegas Properties','Vargas Properties')
    lobbyist[3] = re.sub('Wal-Mart.*$','Wal-Mart', lobbyist[3])
    lobbyist[3] = re.sub('Walton Stacy.*$', 'Walton Stacy Partners', lobbyist[3])
    lobbyist[3] = re.sub('Western Rim Invest.*$','Western Rim Investment Advisors', lobbyist[3])
    lobbyist[3] = lobbyist[3].replace('Whitfield, Marcus','Marcus Whitfield')
    lobbyist[3] = lobbyist[3].replace("\x19","'")
    lobbyist[3] = lobbyist[3].rstrip()

canonClients = {}
done = []
# canonClients[83] = ("Scarbrough Ventures", "Property Owner")
# canonClients[736] = ("Lack and Hurley", "Property Owner")

for lobbyist in reversed(lobbyists):
    if lobbyist[5] not in canonClients:
        # print("haven't seen " + lobbyist[5] + " before.")
        canonClients[lobbyist[5]] = (lobbyist[3], lobbyist[4])
    else:
        # no longer excluding my handcrafted list of "duplicate" sets that make no sense
        # if lobbyist[5] not in (56, 68, 70, 72, 98, 106, 110, 117, 119, 121, 143, 162, 203, 299, 300, 307, 319, 362, 458, 460, 492, 675, 684, 719, 722, 727, 740, 743, 763, 817, 818) and lobbyist[3] not in ("Radec Management",):
            if lobbyist[3] != canonClients[lobbyist[5]][0] or lobbyist[4] != canonClients[lobbyist[5]][1]:
                if lobbyist[5] not in done:
                    done.append(lobbyist[5])
                    print("changing " + lobbyist[3] + " to " + str(canonClients[lobbyist[5]][0]) + " because lobbyist[5] is " + str(lobbyist[5]))
                lobbyist[3] = canonClients[lobbyist[5]][0]
                lobbyist[4] = canonClients[lobbyist[5]][1]


from fuzzywuzzy import fuzz
import re

# Wrangling lobbyist names

for lobbyist in lobbyists:
    lobbyist[2] = lobbyist[2].rstrip()
    lobbyist[2] = lobbyist[2].replace('  ',' ')
    lobbyist[2] = lobbyist[2].replace('"Trey S','"Trey" S')
    lobbyist[2] = lobbyist[2].replace('"Smitty ','"Smitty" ')
    lobbyist[2] = lobbyist[2].replace('Salda?a','Saldaña')
    lobbyist[2] = re.sub('Amelia Lopez$','Amelia Lopez Saltarelli',lobbyist[2])
    lobbyist[2] = lobbyist[2].replace('Bob Digneo','Robert Digneo')
    lobbyist[2] = lobbyist[2].replace('Gerbracht Heidi','Heidi Gerbracht')
    lobbyist[2] = lobbyist[2].replace('Wunsch Karen','Karen Wunsch')
    
uniqueL = []
for lobbyist in lobbyists:
    if lobbyist[2] not in uniqueL:
        uniqueL.append(lobbyist[2])
    
for person in uniqueL:
    for other in uniqueL:
        if person != other and fuzz.token_set_ratio(person, other) > 90:
            print(other + ' changed to ' + person)
            for lobbyist in lobbyists:
                if lobbyist[2] == other:
                    lobbyist[2] = person
            uniqueL.remove(other)

            '''
    lobbyist[2] = lobbyist[2].replace('Katie King Ogden','Katie Ogden')
    lobbyist[2] = lobbyist[2].replace('Bustamante Christine','Christine Bustamante')
    
    '''

# below is just to look at what values are in the list

newL = []
for lobbyist in lobbyists:
    if lobbyist[2] not in newL:
        newL.append(lobbyist[2])
newL.sort()   
print(newL)

# I'm hesitant to use the industry field in any way.

for lobbyist in lobbyists:
    lobbyist[4] = lobbyist[4].rstrip()
    lobbyist[4] = lobbyist[4].replace('  ',' ')
    lobbyist[4] = re.sub(' Corporation$','', lobbyist[4])
    lobbyist[4] = lobbyist[4].replace("\x19","'")
    
uniqueI = []
for lobbyist in lobbyists:
    if lobbyist[4] not in uniqueI:
        uniqueI.append(lobbyist[4])

uniqueI.sort()   
print(uniqueI)

# checking to see if the dates are strings or datetimes right now.

q = type(lobbyists[1][0])
print(q)
print(lobbyists[1][0])

print("lobbyists " + str(len(lobbyists)))

# This is to get rid of multiple entries for the same person hired to rep related entities on the same day.

newl = []
for i in lobbyists:
    if i not in newl:
        newl.append(i)

print("newl " + str(len(newl)))

for first in newl:
    for second in newl:
        if first[:4] == second[:4]: # and first[4] != second[4]:
            if "Zoning" in first[4] or "Property" in first[4]:
                first[4] = second[4]
            newl.remove(second)

print("newl reduced to " + str(len(newl)))

extensions = 0
# samestart = 0

print(newl[1])

# The next block is intended to delete entries that represent the same lobbyist merely renewing a registration.
# I'm assuming that the same representation is continuing if there's a gap of less than 180 days in 
# the lobbyist's registration. I picked the number 180 out of thin air.

for first in newl:
    for second in newl:
        if first[2:4] == second[2:4] and first[0] <= second[0] and first[1] < second[1] and (first[1] + datetime.timedelta(days=180)) > second[0]:
            # print("for " + first[2] + " " + first[3] + " extending " + str(first[1]) + " to " + str(second[1]))
            first[1] = second[1]
            extensions = extensions + 1
            newl.remove(second)

print('deleted ' + str(extensions) + ' extensions.')

# In the past, this step created entries that were identical except the end date. This probably had something to do
# with the bug where lobbyists counts were too high for some clients like Riverside Resources and Met Center.

# Import industries_standardized.csv and join new columns from that file.

file = './data/industries_standardized.csv'
industries = {}

newFile = open(file)
newReader = csv.reader(newFile)
newData = list(newReader)
# print(newData)
for row in newData[1:]:
    industries[row[1]] = (row[2],row[3])

print(newl[1])

for row in newl:
    if row[4] in industries:
        row.append(industries[row[4]][0])
        row.append(industries[row[4]][1])
    else:
        row.extend([None,None])
        
print(newl[211])
    

# putting data in pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lobbyFrame = pd.DataFrame(newl, columns=('Start', 'End', 'Lobbyist', 'Client', 'Industry', 'Cluster', 'Address', 'Confidence', 'IndustryCat1','IndustryCat2'))

lobbyFrame.describe()

lobbyFrame.sort_values(by='Start')
lobbyFrame.head(10)

lastDay = lobbyFrame['Start'].max() # This is the end date of the line chart!!
print(lastDay)

# making a new output file:

outputFile = open('./data/lobbyistTimeRange.csv', 'w', newline='')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['Start', 'End', 'Lobbyist', 'Client', 'Industry', 'Cluster','Address','Confidence','IndustryCat1','IndustryCat2'])
for row in newl:
    outputWriter.writerow(row)
outputFile.close()
        

lobbyFrame['Client'].value_counts()

'''
I'm trying to find every group of rows with the same Date and Client, and replace them with a single row with the
sum of the old Change values as the new Change value.
    '''

changes = []
for lobbyist in newl:
    # start date, client name, add one
    changes.append([lobbyist[0], lobbyist[3], 1])
    changes.append([lobbyist[1], lobbyist[3], -1])


cumFrame = pd.DataFrame(changes, columns=('Date', 'Client', 'Change'))
# cumFrame['Date'] = pd.to_datetime(cumFrame.Date)
cumFrame = cumFrame.sort_values('Date')

print(cumFrame.loc[cumFrame['Client'] == 'AT&T'])

cFrame = cumFrame.groupby(('Client','Date')).sum()

cFrame.head(20)

cFrame.reset_index(inplace=True)  
cFrame.head(20)

cFrame['Cumulative'] = cFrame.groupby('Client')['Change'].apply(lambda x: x.cumsum())
# df['no_cumulative'] = df.groupby(['name'])['no'].apply(lambda x: x.cumsum())
cFrame.head(20)

croppedFrame = cFrame[cFrame.Date < lastDay]
# croppedFrame = cumFrame[cumFrame.Date < '2016-01-21']
print(croppedFrame.loc[cFrame['Client'] == 'AT&T'])
croppedFrame.tail()

croppedFrame.describe()

croppedFrame.to_csv('./data/lobbyistByClient.csv', index = False)

