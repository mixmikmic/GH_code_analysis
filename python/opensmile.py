import subprocess
import sys
import os
import os.path
import numpy as np
from scipy.io import arff
sys.path.append('/home/vincent/enschede/explosmile/explosmile')
import load_iemocap as li
from HTK import HTKFile
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Specify paths
opensmile_path = os.path.expanduser("~/openSMILE-2.1.0") # Where is opensmile?
iemocap_path = "/media/vincent/enschede/IEMOCAP_full_release" # location of the IEMOCAP dataset
#OPENSMILE_CONF = os.path.join('config', 'gemaps', 'GeMAPSv01a.conf')
#OPENSMILE_CONF_PATH = os.path.join(OPENSMILE_PATH, OPENSMILE_CONF)

session = "Session1"

# get content of directories
wav_files_path = os.path.join(iemocap_path, session + "/sentences/wav")
wavfiles_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(wav_files_path)) for f in fn]
wavfiles = np.array(wavfiles_list)

label_files_path = os.path.join(iemocap_path, session + "/dialog/EmoEvaluation") # path to txt files with labels
labfiles = os.listdir(label_files_path) # list of filenames

# remove filesnames that do not start with . and end with txt or wav
realwavfiles = li.returnrealfiles(wavfiles)
reallabfiles = li.returnrealfiles(labfiles)

wavfile_index = 0

# Let's focus on one wav and lab file for now
wav_filename = str(realwavfiles[wavfile_index])
wav_fullpath =  os.path.join(wav_files_path,wav_filename )

# find matching label file
lab_fullpath = li.find_matching_label_file(wav_filename, reallabfiles, label_files_path)

# define opensmile command
opensmile_conf = os.path.join(opensmile_path,"config/MFCC12_0_D_A.conf")
htkfile = "/media/vincent/enschede/" + wav_filename.split("/")[-1].replace('.wav', '') + ".htk"
command = "SMILExtract -I {input_file} -C {conf_file} -O {output_file}".format(
                                    input_file=wav_fullpath,
                                    conf_file=opensmile_conf,
                                    output_file=htkfile)

# Run the Open smile command and save output
output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

# load htkfile, credits to https://github.com/danijel3/PyHTK, see license inside code
htk = HTKFile() 
htk.load(htkfile)
# object is a list of lists, now turn into numpy array:
htk_np_data = np.array(htk.data)

print(htk_np_data.shape)

# Duration of utterance in seconds:
htk.nSamples / 100

plt.plot(htk_np_data[:,4])
plt.show()

# load labels and put them in a pandas data.frame
labels = li.readlabtxt(lab_fullpath)

labels.head()

# find the row for which the TURN_NAME matches the wav features file
TURNNAMEwav = htkfile.split("/")[-1].split(".ht")[0]
b = [index for index, item in enumerate(labels['TURN_NAME']) if TURNNAMEwav == item][0]
# extract label for this utterance
label = labels['EMOTION'][b]





