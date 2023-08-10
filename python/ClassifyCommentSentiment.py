import subprocess
import shlex
import os.path
import sys

#######################################################################################################
## Modify the three lines below to make this program work on your computer.                          ##
## Be careful with the direction of the slashes / and include a slash at the end of the second path. ##
#######################################################################################################
SentiStrengthLocation = "D:/Downloads/k-pop test/SentiStrength.jar" #This must point to the location of SentiStrength on your computer
SentiStrengthUnzippedTextFilesLocation = "D:/Downloads/k-pop test/SentiStrength_Data/" #This must point to the location of the unzipped SentiStrength data files on your computer
FileToClassify = "D:/Downloads/k-pop test/BLACKPINK_eng-1kYrp_Bs8DU_commentsOnly.txt" #This must point to the location of the file that you want classified.

#Test file locations and quit if anything not found
if not os.path.isfile(SentiStrengthLocation):
    print("SentiStrength not found at: ", SentiStrengthLocation)
    sys.exit()
if not os.path.isfile(FileToClassify):
    print("File to classify not found at: ", FileToClassify)
    sys.exit()
if not os.path.isdir(SentiStrengthUnzippedTextFilesLocation):
    print("SentiStrength data folder not found at: ", SentiStrengthUnzippedTextFilesLocation)
    sys.exit()

# Test if SentiStrength is working
def RateSentiment(sentiString):
    #open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(shlex.split("java -jar '" + SentiStrengthLocation + "' stdin sentidata '" + SentiStrengthUnzippedTextFilesLocation + "'"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    #Can't send string in Python 3, must send bytes
    b = bytes(sentiString.replace(" ","+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    #convert from byte
    stdout_text = stdout_byte.decode("utf-8") 
    #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1 -5
    stdout_text = stdout_text.rstrip().replace("\t"," ")
    return stdout_text + " " + sentiString

print("If SentiStrength is working then 3 and -1 will be next:", RateSentiment("A lovely day!"),"\n")

print("Running SentiStrength on file " + FileToClassify + " with command:")
cmd = 'java -jar "' + SentiStrengthLocation + '" sentidata "' + SentiStrengthUnzippedTextFilesLocation + '" input "' + FileToClassify + '"'
print(cmd, "\n")
p = subprocess.Popen(shlex.split(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
wait =input("Finished! The results will be in a file with a name from\n" + FileToClassify + "\nending in out.txt.\nPress enter to continue.")

import subprocess
import shlex
import os.path
import sys

#######################################################################################################
## Modify the three lines below to make this program work on your computer.                          ##
## Be careful with the direction of the slashes / and include a slash at the end of the second path. ##
#######################################################################################################
SentiStrengthLocation = "D:/Downloads/k-pop test/SentiStrength.jar" #This must point to the location of SentiStrength on your computer
SentiStrengthUnzippedTextFilesLocation = "D:/Downloads/k-pop test/SentiStrength_Data/" #This must point to the location of the unzipped SentiStrength data files on your computer
FolderOfFilesToClassify = "D:/Downloads/k-pop test/TWICE-BTS-EXO-BLACKPINK_english comments, one file per video/" #This must point to the location of the folder of files that you want classified.

#Test file locations and quit if anything not found
if not os.path.isfile(SentiStrengthLocation):
    print("SentiStrength not found at: ", SentiStrengthLocation)
    sys.exit()
if not os.path.isdir(FolderOfFilesToClassify):
    print("Folder of files to classify not found at: ", FolderOfFilesToClassify)
    sys.exit()
if not os.path.isdir(SentiStrengthUnzippedTextFilesLocation):
    print("SentiStrength data folder not found at: ", SentiStrengthUnzippedTextFilesLocation)
    sys.exit()

# Test if SentiStrength is working
def RateSentiment(sentiString):
    #open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(shlex.split("java -jar '" + SentiStrengthLocation + "' stdin sentidata '" + SentiStrengthUnzippedTextFilesLocation + "'"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    #Can't send string in Python 3, must send bytes
    b = bytes(sentiString.replace(" ","+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    #convert from byte
    stdout_text = stdout_byte.decode("utf-8") 
    #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1 -5
    stdout_text = stdout_text.rstrip().replace("\t"," ")
    return stdout_text + " " + sentiString

print("If SentiStrength is working then 3 and -1 will be next:", RateSentiment("A lovely day!"))

print("\nRunning SentiStrength with command")
cmd = 'java -jar "' + SentiStrengthLocation + '" sentidata "' + SentiStrengthUnzippedTextFilesLocation + '" annotateCol 1 inputFolder "' + FolderOfFilesToClassify + '" overwrite'
print(cmd, "\n")

p = subprocess.Popen(shlex.split(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
wait =input("\nPlease wait for the black window to vanish - may take 15 minutes!\nDO NOT RUN THIS AGAIN ON THE SAME FILES.\nThe results will be added to each file in\n" + FolderOfFilesToClassify + ".\n")

