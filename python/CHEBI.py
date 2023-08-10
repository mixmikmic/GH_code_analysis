from bioservices import chebi
#import libchebipy

instance = chebi.ChEBI()

#open file
f = open('drugs.txt')
druglist = []
for lines in f:
    druglist.append(lines)
f.close()

nameList = {} #keys = drug name; value = idList
# https://pythonhosted.org/bioservices/_modules/bioservices/chebi.html
# http://libchebi.github.io/libChEBI%20API.pdf
molfile = open('alldrug.sdf', 'w')
for drug in druglist:
    result = instance.getLiteEntity(drug)
    if len(result) == 0:
        print(drug)
        nameList[drug] = {}
    else:
        idList = {} #keys = query name; value = ID
        
        for query in range(len(result)):
            idList[result[query][1]] = result[query][0] #save id in idList
            realID = result[query][0][6:]
            #opt out to get molfile and write to file named with its ID
            StructureInfo = instance.getUpdatedPolymer(realID)
            if StructureInfo != None:
                print(realID, file = molfile)
                print(StructureInfo['updatedStructure'], file = molfile)
                print('>  <ACTIVITY>\n5.00\n$$$$',file=molfile)
            else:
                print(realID, 'None')
        nameList[drug] = idList

molfile.close()

       


instance.getUpdatedPolymer('85989') == None




nameList

instance.getUpdatedPolymer(28417)



