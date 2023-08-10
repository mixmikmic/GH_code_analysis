#If you are running this in IBM Apache Spark (via Data Science Experience)
base_url = 'https://dal05.objectstorage.service.networklayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b'

#ELSE, if you are outside of IBM:
base_url = 'https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b'

#NOTE: using the 2nd base_url, if you are outside of IBM, will be slower. :/

#Defining a local data folder to dump data

import os

mydatafolder = os.path.join( os.environ['PWD'], 'my_data_folder' )
if os.path.exists(mydatafolder) is False:
    os.makedirs(mydatafolder)

import os

basic_container = 'simsignals_v3_zipped'
basic4_zip_file = 'basic4.zip'

os.system('curl {}/{}/{} > {}'.format(base_url, basic_container, basic4_zip_file, mydatafolder + '/' + basic4_zip_file))

get_ipython().system('ls -al my_data_folder/basic4.zip')

filename = 'primary_small_v3.zip'
primary_small_url = '{}/simsignals_v3_zipped/{}'.format(base_url, filename)
os.system('curl {} > {}'.format(primary_small_url, mydatafolder +'/'+filename))

filename = 'public_list_primary_v3_small_21june_2017.csv'
primary_small_csv_url = '{}/simsignals_files/{}'.format(base_url, filename)
os.system('curl {} > {}'.format(primary_small_csv_url, mydatafolder +'/'+filename))

med_N = '{}/simsignals_v3_zipped/primary_medium_v3_{}.zip'

for i in range(1,6):
    med_url = med_N.format(base_url, i)
    output_file = mydatafolder + '/primary_medium_v3_{}.zip'.format(i)
    print 'GETing', output_file
    os.system('curl {} > {}'.format(med_url, output_file ))

filename = 'public_list_primary_v3_medium_21june_2017.csv'
med_csv_url = '{}/simsignals_files/{}'.format(base_url, filename)
os.system('curl {} > {}'.format(med_csv_url, mydatafolder +'/'+filename))

filename = 'public_list_primary_v3_full_21june_2017.csv'
prim_full = '{}/simsignals_files/{}'.format(base_url, filename)
os.system('curl {} > {}'.format(prim_full, mydatafolder +'/'+filename))

import requests
import copy

file_list_container = 'simsignals_files'
file_list = 'public_list_primary_v3_full_21june_2017.csv'
primary_data_container = 'simsignals_v3'

r = requests.get('{}/{}/{}'.format(base_url, file_list_container, file_list), timeout=(9.0, 21.0))
filecontents = copy.copy(r.content)

full_primary_files = [line.split(',') for line in filecontents.split('\n')]
full_primary_files = full_primary_files[1:-1] #strip the header and empty last element
full_primary_files = map(lambda x: x[0]+".dat", full_primary_files)  #now list of file names (<uuid>.dat)

#save your data into a local subfolder
save_to_folder = mydatafolder + '/primary_data_set'
if os.path.exists(save_to_folder) is False:
    os.mkdir(save_to_folder)

count = 0
total = len(full_primary_files)
for row in full_primary_files:
    r = requests.get('{}/{}/{}'.format(base_url, primary_data_container, row), timeout=(9.0, 21.0))
    
    if count % 100 == 0:
        print 'done ', count, ' out of ',  total
    count += 1
    
    with open('{}/{}'.format(save_to_folder, row), 'w' ) as fout:
        fout.write(r.content)

## Using Spark -- can parallelize the job across your worker nodes
import ibmseti
def retrieve_and_process(row):
    try:
        r = requests.get('{}/{}/{}'.format(base_url, primary_data_container, row), timeout=(9.0, 21.0))
    except Exception as e:
        return (row, 'failed', [])
    
    aca = ibmseti.compamp.SimCompamp(r.content)
    spectrogram = aca.get_spectrogram() # or do something else
    features = my_feature_extractor(spectrogram) #example external function for reducing the spectrogram into a handful of features, perhaps
    
    signal_class = aca.header()['signal_classifiation']
        
    return (row, signal_class, features)

npartitions = 60  
rdd = sc.parallelize(full_primary_files, npartitions)

#Now ask Spark to run the job
process_results = rdd.map(retrieve_and_process).collect()



