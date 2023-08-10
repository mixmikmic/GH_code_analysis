# import appropriate packages
import urllib.request
import scipy.io as sio
import pickle
import os

# URLS of the sample dataset
ip_urls = list()
ip_urls.append("http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat")
ip_urls.append("http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat")
ip_urls.append("http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat")

ip_names = list()
ip_names.append('Indian_pines.mat')
ip_names.append('Indian_pines_corrected.mat')
ip_names.append('Indian_pines_gt.mat')

data_names = list()
data_names.append('indian_pines')
data_names.append('indian_pines_corrected')
data_names.append('indian_pines_gt')

print("--- downloading with urllib2")

for (url, name) in zip(ip_urls, ip_names):
    f = urllib.request.urlopen(url)
    data = f.read()
    with open(name, "wb") as code:
        code.write(data)

# create an empty dictionary to hold the data 
hsi_data = {}

# loop through key and filename
for name, file in zip(data_names, ip_names):
    
    # import .mat file
    temp = sio.loadmat(file)
    
    # save hsi data in dictionary
    hsi_data[name] = temp[name]

# Save data in pickle file
with open('indianpines.pickle', 'wb') as handle:
    pickle.dump(hsi_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# check if the files are the same
with open('indianpines.pickle', 'rb') as handle:
    temp = pickle.load(handle)
    
for key in hsi_data.keys():
    
    verdict = hsi_data[key].all()==temp[key].all()
    print('Data w/ key {k} is the same: {t}'.format(k=key, t=verdict))

# remote data from directory



