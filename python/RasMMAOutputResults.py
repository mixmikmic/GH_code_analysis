tag = "parite_0.8" # used for pickle name
outputPath = "output/RasMMA/" + tag + "/parite/"
pickleDir = outputPath + "pickle/"
# pickleDir = outputPath

import pickle

# read the results from pickle files
with open(pickleDir + tag + '_intermediate.pickle', 'rb') as handle:
    intermediate = pickle.load(handle)
with open(pickleDir + tag + '_initialDict.pickle', 'rb') as handle:
    initialDict = pickle.load(handle)
with open(pickleDir + tag + '_roundInfos.pickle', 'rb') as handle:
    roundInfos = pickle.load(handle)

# calculate motif lengths of all common motifs
def getMotifsLengthList(motifs):
    motifLens = list()
    for motif in motifs:
        startIdx =motif[1]
        endIdx = motif[2]
        mLen = endIdx - startIdx + 1
        motifLens.append(mLen)
    return motifLens

def findGeneratedRoundNumber(clusterName, roundInfosDict):
    for key, value in roundInfosDict.items():
        if clusterName in value:
            return key
    return -1

import csv

descendant_dict = dict()
groupInfo_list = list()
groupMotif_dict = dict()

intermediate_list = sorted(intermediate.items(), key=lambda x : x[0])
for item in intermediate_list:
    value = item[1] # get original dict value
    score = value[0]
    clusterName = value[1][0]
    memberSet = value[2]
    motifs = value[1][1]
    
    # calculate motif lengths of all common motifs
    motifsLens = getMotifsLengthList(motifs) # is a list of numbers
    totalMotifLen = sum(motifsLens) # sum the list

    motifsCount = len(motifs)
    
    descendants = set()
    for member in memberSet:
        if member[0] == "G":
            for descendant in descendant_dict[member]:
                descendants.add(descendant)
        else:
            descendants.add(member)
    descendant_dict[clusterName] = descendants
    
    
    groupMotif_dict[clusterName] = motifs
    roundNumber = findGeneratedRoundNumber(clusterName, roundInfos)
    groupInfo_list.append((roundNumber, clusterName, score, memberSet, motifsCount, motifsLens, totalMotifLen))

with open(pickleDir + tag + "_descendant.pickle", 'wb') as f:
    pickle.dump(descendant_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# write file "GroupInfo.csv" :  clusterName, score, members, motifCount, common motifs length list
with open(outputPath + tag + "_GroupInfo.csv", 'w', newline='') as infoFile:
    spamwriter = csv.writer(infoFile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["Round", "ClusterName", "SimilarityScore", "Members", "MotifsCount", "Motifs_Length", "Total_MotifLength"]
    spamwriter.writerow(header)
    
    # write initial cluster informations(i.e., hooklogs)
    for key in sorted(initialDict.keys(), key = lambda x : int(x[1::])):
        # something like this: (0, "G1", "N/A", "abc", 1, 109)
        originDataRow = (0, key, "N/A", initialDict[key][0], 1, initialDict[key][1], initialDict[key][1])
        spamwriter.writerow(originDataRow)
        
    # write cluster informations
    for group in groupInfo_list:
        spamwriter.writerow(group)
        
with open(outputPath + tag + "_Descendants.csv", "w", newline='') as descFile:
    spamwriter = csv.writer(descFile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["ClusterName", "Descendant Counts", "Descendants"]
    spamwriter.writerow(header)
    for key in sorted(descendant_dict.keys(), key = lambda x : int(x[1::])):
        row = (key, len(descendant_dict[key]), descendant_dict[key])
        spamwriter.writerow(row)
        
# write file "Motifs.csv" :  clusterName, MotifNumber, apis
with open(outputPath + tag + "_Motifs.csv", 'w', newline='', encoding='utf-8') as motifFile:
    spamwriter = csv.writer(motifFile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["ClusterName", "MotifIndex", "MotifLength", "Common Motif APIs"]
    spamwriter.writerow(header)

    for key in sorted(groupMotif_dict.keys(), key = lambda x : int(x[1::])):
        group_motifs = groupMotif_dict[key]
        motifIdx = 0
        for motif in group_motifs:
            firstMotifAPI = True
            motifLen = len(motif[0])
            for api in motif[0]:
                if(firstMotifAPI):
                    row = (key, motifIdx, motifLen, api)
                    firstMotifAPI = False
                else:
                    row = ("", "", "", api)
                spamwriter.writerow(row)
            motifIdx += 1
            

# output residual information of SBBGCA

with open(pickleDir + tag + '_residual.pickle', 'rb') as handle:
    residual = pickle.load(handle)
    
with open(outputPath + tag + "_GroupInfo.csv", 'a', newline='') as expandGroupInfo:
    spamwriter = csv.writer(expandGroupInfo, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["ClusterName", "Members", "MotifLength"]
    
    spamwriter.writerow("")
    spamwriter.writerow(("Residual Clusters:","",""))
    spamwriter.writerow(header)
    
    for key, value in residual.items():
        clusterName = value[0][0]
        motifsList = value[0][1]
        motifLens = getMotifsLengthList(motifsList)
        members = value[1]
        if( len(members) == 0 ):
            row = (clusterName, "N/A", motifLens)
        else:
            row = (clusterName, members, motifLens)
            
        spamwriter.writerow(row)



