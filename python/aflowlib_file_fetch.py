import json
import urllib
from urllib.request import urlopen
import time
import os

#from urllib.request import urlopen # preamble

SERVER='http://aflowlib.duke.edu' # server name
PROJECT='AFLOWDATA/ICSD_WEB' # project name

BRAVAIS = ['CUB','FCC',
           'BCC','TET',
           'BCT','ORC',
           'ORCF','ORCI',
           'ORCC','HEX',
           'RHL','MCL',
           'MCLC','TRI']

ibravais = 13

URL=SERVER+'/'+PROJECT+'/'+BRAVAIS[ibravais]

url_request = urlopen(URL+'?aflowlib_entries&format=json').read().decode('utf-8')
entry=json.loads(url_request) # load

root_folder = 'full_file_stack'+'_'+BRAVAIS[ibravais]
os.mkdir(root_folder)
#os.chdir(root_folder)

def fetch_files(compound, root_folder, lfiles):
    
    folder = './'+root_folder+'/'+compound
    os.mkdir(folder)
    os.chdir(folder)
    
    for f in lfiles:
        print(f,URL+'/'+compound+'/'+f)
        urllib.request.urlretrieve(URL+'/'+compound+'/'+f,f)
    
    os.chdir('../../')
    
    time.sleep(1.0)

aflowlib_entries = entry['aflowlib_entries']
print(aflowlib_entries)

def select_files(lfiles,selection):
    file_selec = []
    for s in selection:
        file_selec = file_selec + [f for f in lfiles if s in f]
    
    #print(file_selec)
    return file_selec

selection = ['POSCAR','CONTCAR','EIGENVAL','KPOINTS','OUTCAR','INCAR','cif']

for index in aflowlib_entries:
    
    compound = aflowlib_entries[index]
    urlentry=URL+'/'+compound+'/'+'?format=json'
    #urlentry_files = URL+'/'+c+'/'+'?files'
    aflow_entry=json.loads(urllib.request.urlopen(urlentry).read().decode('utf-8'))

    lfiles = aflow_entry['files']
    #print(lfiles)
    sfiles = select_files(lfiles,selection)
    print(sfiles)
    fetch_files(compound, root_folder, sfiles)

urlentry
t=json.loads(urllib.request.urlopen(urlentry).readall().decode('utf-8')) 
#for c in aflowlib_entries:

print(urlentry_files)
#print(t['aurl'])
t['files']

urllib.request.urlopen(urlentry_files).readall().decode('utf-8')

for f in t['files']:
    print(f,URL+'/'+c+'/'+f)
    urllib.request.urlretrieve(URL+'/'+c+'/'+f,f)

