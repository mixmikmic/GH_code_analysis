import pickle
import os
import shutil

dataPickle = "Report/0.1.pickle"
family_root_dir = "tracelogs_by_family_Temu20/"
hkPoolDir = "hooklogs/temu_20/"
# read the target pickle files
with open(dataPickle, 'rb') as handle:
    dataContent = pickle.load(handle)

for famName, md5s in dataContent.items():
    familyDir = family_root_dir + famName
    
    for root, dirs, files in os.walk(hkPoolDir):
        for fEntry in files:
            path = root + fEntry
            md5 = fEntry.split("_")[0]
            if md5 in md5s:
                if not os.path.isdir(familyDir): os.makedirs(familyDir)
                shutil.copyfile(path, familyDir+"/"+fEntry)

# dataContent['bettersurf']

# Key API Name = CreateProcess
# Key Attribute = dwProcessId

def getSampleRelation(familyPath, sampleMD5Dict):
    keyAPI = "CreateProcess"
    keyAttribute = "dwProcessId"
    
    possibleRoot = set()
    result = dict()
    md5RelationDict = dict() # a dict {key=main: value=child_pid} (or key=child, value=grandchild_pid)
    
    for md5 in sorted(sampleMD5Dict.keys()):
        traceFiles = sampleMD5Dict[md5]
        if len(traceFiles) == 1: # if single file, skip it.
            possibleRoot.add(traceFiles[0])
            continue

        for trace in traceFiles: # trace all files if they have same md5
            
            handle = open(familyPath + trace, 'rb')
            child = list()
            while(1):
                line = handle.readline().decode("ISO 8859-1").strip() # MIKE: 20170616, for python 3
                if not line: 
                    break
                if(line[0] is '#'):
                    api = handle.readline().decode("ISO 8859-1").strip() # see api name
                    if(api == keyAPI):
                        terminateCtr = 0
                        
                        # dwProcessId may appear in createProcess's params, amounts 14 lines
                        while(terminateCtr<14):
                            newLine = handle.readline().decode("ISO 8859-1").strip()
                            if(newLine[0:6] == "Return"):
                                if(newLine.split('=')[1] != "SUCCESS"): break

                            if(newLine[0:11] == keyAttribute):
                                child.append(newLine.split('=')[1])
                                break

                            terminateCtr+=1 # defend of infinite loop
                            
            md5RelationDict[trace] = child

    
    for trace in sorted(md5RelationDict.keys()):
        childList = md5RelationDict[trace]

        if trace not in result.keys():
            traceList = list()
            for child in childList:
                hashValue = trace.split("_")[0]
                fName = hashValue + "_" + child + ".trace.hooklog"

                if fName in sampleMD5Dict[hashValue]:
                    traceList.append(fName)
#                 else:
#                     print(trace, child)
            result[trace] = traceList
        else:
            print("!!! - ",trace)
    
    reverseRelation = dict()
    for k in sorted(result.keys()):
        v = result[k]
        if v:
            for ele in v:
                if ele in reverseRelation.keys(): print(ele,";",k)
                reverseRelation[ele] = k

    possibleRoot.update(set(md5RelationDict.keys()))
    totalTraceCount = len(possibleRoot)
    
    for k in reverseRelation.keys():
        if k in possibleRoot:
            possibleRoot.remove(k)
                        
    return result, possibleRoot, totalTraceCount

# Move all other main processes which didn't fork any child to mainDir
import os
def moveOtherMainProcs(familyPath, mainDir):
    files = os.listdir(familyPath)
    for mainProc in files:
        if os.path.isfile(familyPath+mainProc):
            shutil.move(familyPath+mainProc, mainDir)

import shutil
def separateProcessByGeneration(familyPath, levels):
    print(familyPath)
    
    for level, malwrs in levels.items():
        if level == 1:
            myDir = familyPath + 'main/'
        elif level == 2:
            myDir = familyPath + 'child/'
        else:
            myDir = familyPath + str(level) + ' child/'
            
        if not os.path.isdir(myDir): os.makedirs(myDir)
            
        for mal in malwrs:
            shutil.move(familyPath+mal, myDir)

familyPath = 'tracelogs_by_family_Temu20/razy'

if os.path.isdir(familyPath):
    sampleMD5Dict = dict() # A dict which key=md5, value=md5_pid.trace

    for root, dirs, files in os.walk(familyPath):
        for fEntry in files:
            if(fEntry == '.DS_Store'): continue # MacOS file system file.

            md5 = fEntry.split("_")[0]
            
            if sampleMD5Dict.get(md5): # classifying traces by md5
                sampleMD5Dict[md5].append(fEntry)
            else:
                sampleMD5Dict[md5] = [fEntry]
    print("MD5 kinds: ", len(sampleMD5Dict.keys()))
    print("Have multi-procs md5:")
    ctr = 0
    for key, value in sampleMD5Dict.items():
        if len(value) > 1:
            ctr+=1
#             print(key, len(value))
    print("multi-process samples count:", ctr)
            
    result, possibleRoot, totalTraceCount = getSampleRelation(familyPath + '/' , sampleMD5Dict)

levels = dict()
currentLevel = 1

levels[currentLevel] = possibleRoot
classifiedTraceCount = len(possibleRoot)
print("currentLevel:",currentLevel , " - ", len(levels[currentLevel]))
while(classifiedTraceCount < totalTraceCount):
    levels[currentLevel+1] = set()
    
    for parent in levels[currentLevel]:
#         print(parent, result[parent])
        
        if parent in result.keys():
            for t in result[parent]:
                levels[currentLevel+1].add(t)
                classifiedTraceCount+=1
        else:
            pass
    currentLevel+=1
    print("currentLevel:",currentLevel , " - ", len(levels[currentLevel]))
    

tl = 0
for lv, eles in levels.items():
    print("Level:", lv, " - 個數:", len(eles))
    tl += len(eles)
print("總共 ", tl, " 個")

# Do generation separation
separateProcessByGeneration(familyPath + '/', levels)



