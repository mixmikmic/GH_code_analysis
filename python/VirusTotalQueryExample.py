# Call 'VirusTotalQuery.ipynb' to get reports
get_ipython().run_line_magic('run', 'VirusTotalQuery.ipynb')

import json
from time import sleep

# main function
dataRootPath = "./hooklogs/dataSet" # set your hooklogs path
outputDirPath = "./VTReport/" # set your output dir path
requestLimit = 300 # virusTotal request limit quota per minute
noReportList_fileName = "noReport_list.txt" # fileName of noReportList

hash_list = listMalwrHash(dataRootPath)

counter = 0 # use for VirusTotal maximum requests limit
noReport_list = [] # storing no report Hash

# query loop
for malwrHash in hash_list :
    if counter == requestLimit:
        sleep(50) # sleep 50secs
        counter=0
    
    report = queryReport(malwrHash)
    
#     Check whether report in VT
    if report["response_code"] != 1: # 1 => VT has report in DB
        noReport_list.append((report["resource"], report["response_code"], report["verbose_msg"]))
    else:
        if not os.path.exists(outputDirPath):
            os.makedirs(outputDirPath)
        filePath = os.path.join(outputDirPath, malwrHash + ".txt")

        reportFile = open(filePath, "w")
        json.dump(report, reportFile, sort_keys = True ,indent = 4) # dump json struct. (like dict[])
        reportFile.close()

        counter+=1

    
# output no report list
if not os.path.exists(outputDirPath):
            os.makedirs(outputDirPath)

noReport_list_Path = os.path.join(outputDirPath, noReportList_fileName)
with open(noReport_list_Path, 'w') as errOutFile:
    for recordRow in noReport_list:
        errOutFile.write(" ".join( str(item) for item in recordRow) + "\n")

