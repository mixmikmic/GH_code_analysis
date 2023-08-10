get_ipython().run_line_magic('run', 'CollectForestInfo.ipynb')
import csv
import os
import shutil

analyzeFamilyName = "bettersurf"
generation = "main"

tag = analyzeFamilyName + "_0.8" # used for pickle name
outputPath = "output/RasMMA/" + tag + "/" + generation + "/"
pickleDir = outputPath + "pickle/"

intermediatePicklePath = pickleDir + tag + '_intermediate.pickle'
residualPicklePath = pickleDir + tag + '_residual.pickle'
        
forestInfo = CollectForestInfo(intermediatePicklePath,
                               residualPicklePath,
                               True) # one pickle is a forest

treeCount = forestInfo.getTreeRootCount()
malwrCount = forestInfo.getForestMemberCount()
print("Behavior trees: ", treeCount)
print("process in trees: ", malwrCount)

mems = forestInfo.getForestMembers()
# print(forestInfo.getForestMemberCount())

fuck = dict()
for m in mems:
    name = m.split("_")[0]
    if name not in fuck.keys():
        fuck[name] = 1
    else:
        fuck[name] += 1

c = dict()
for k, v in fuck.items():
    if v not in c.keys():
        c[v] = 1
    else:
        c[v] += 1
print(c)
nameSet = {m.split("_")[0] for m in mems}
print("analyzing: ",len(nameSet))

repAPISeq_dict = forestInfo.getRepAPISeq_dict()
# repAPILenList = [len(repAPI) for tree, repAPI in repAPISeq_dict.items()]

repAPILenList = list()
trMemCountList = list()
for tree, repAPI in repAPISeq_dict.items():
    repAPILenList.append(len(repAPI))
    trMemCountList.append(len(forestInfo.getTreeMembers(tree)))
    
opStr = "{"
for l in range(len(repAPILenList)):
    if l == len(repAPILenList) -1:
        opStr += str(repAPILenList[l]) + "}"
    else:
        opStr+=str(repAPILenList[l]) + "," 
print(opStr)
opStr2 = "{"
for l in range(len(trMemCountList)):
    if l == len(trMemCountList) -1:
        opStr2 += str(trMemCountList[l]) + "}"
    else:
        opStr2+=str(trMemCountList[l]) + "," 
print(opStr2)

trMems = forestInfo.getTreeMembers_dict()
sampleCount = set()
procCount = set()
labeled = {"G186" , "G205", "G220", "G228"} #108 98
for tr, mems in trMems.items():
#     print(mems)
    if tr in labeled:
#         print(mems)
        for m in mems: sampleCount.add(m.split("_")[0])
        procCount.update(mems)
    print("Tree:",tr)
    print("Member size:", len(mems))
#     print("API Len", len(repAPISeq_dict[tr]))
#         print(repAPISeq_dict[tr])

print("Labeled Sample:" , len(sampleCount))
print("Labeled Processes:", len(procCount))

get_ipython().run_line_magic('run', 'FeatureTrace.ipynb')
from openpyxl import load_workbook, Workbook

data_dir = "tracelogs_analysis_temu20/" + analyzeFamilyName + "/" + generation + "/"

treeMember_dict = forestInfo.getTreeMembers_dict()
baseline_dict = dict()

for tree in forestInfo.getTreeRootNameList():
    members = treeMember_dict[tree]
    for file in os.listdir(data_dir):
        shortName = file.split("_")[0][0:6]
        pid = file.split("_")[1].split(".")[0]
        nickname = shortName+"_"+pid
        
        if nickname in members:
            baseline_dict[tree] = file
            break
print("Tree \t Base")
for k, v in baseline_dict.items():
    print(k,":", v)

import re
FTrace = FeatureTrace # get class
ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        
# output to excel
wb = Workbook()
ws = wb.active

for tree in sorted(baseline_dict.keys()):
    baseline = baseline_dict[tree]
    repAPI = forestInfo.getRepAPISeq(tree) # get tree rep API seq list
    featureTrace = FTrace(data_dir + baseline).getTrace_noContainTS() # get feature Trace
    originalTrace = FTrace(data_dir + baseline).getOriginalTrace_withoutTS() # get original Trace (selected Params)
    
    repToOrigin = list() # result
    breakPoint = 0
    # compare the api in rep and featureTrace -> find index used for original Trace
    for api in repAPI:
        for index in range(breakPoint, len(featureTrace)):
            featureAPI = featureTrace[index]
            
            if api == featureAPI:
                repToOrigin.append(originalTrace[index])
                breakPoint = index + 1
                break
                
    trMembers = forestInfo.getTreeMembers(tree)
    samples = {member.split("_")[0] for member in trMembers}
    
    # output to excel
    ws.append([tree, "Rep API Sequence Length:"+str(len(repAPI)),
               "Covered Samples:"+str(len(samples)), "Covered Processes:"+str(len(trMembers))])
    
    for index in range(len(repToOrigin)):
        data = ILLEGAL_CHARACTERS_RE.sub(r'', repToOrigin[index])
        ws.append([index+1, data])
        
        
wb.save(outputPath + tag + '_origin.xlsx')

data_dir = "tracelogs_analysis_temu20/" + analyzeFamilyName + "/" + generation + "/" # total sample dir of family
target_dir_base = "tracelogs_analysis_temu20/Selected/"+ analyzeFamilyName + "/" + generation + "/" # target dir

for file in os.listdir(data_dir):
    shortName = file.split("_")[0][0:6]
    pid = file.split("_")[1][0:4]
    nickname = shortName+"_"+pid
    
    treeMemberDict = forestInfo.getTreeMembers_dict()
    for treeID, members in treeMemberDict.items():
        target_dir = target_dir_base + treeID + '/'
        if not os.path.isdir(target_dir): os.makedirs(target_dir)
        
        if nickname in members:
            shutil.copy(data_dir + file, target_dir + file)

#     if nickname in forestInfo.getForestMembers():
#         if not os.path.isdir(target_dir_base): os.makedirs(target_dir_base)
#         shutil.copy(data_dir+file, target_dir_base + file)



