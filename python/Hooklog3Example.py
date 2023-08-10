# if using Hooklog3
get_ipython().run_line_magic('run', 'Hooklog3.ipynb')
Hooklog = Hooklog3

# elis using NestedHooklog3
#%run NestedHooklog3.ipynb
#Hooklog = NestedHooklog3

# elis using FeatureHooklog3
#%run FeatureHooklog3.ipynb
#Hooklog = FeatureHooklog3

# elis using NestedFeatureHooklog3
#%run NestedFeatureHooklog3.ipynb
#Hooklog = NestedFeatureHooklog3

# input 
in_directory = "C:/Users/hsiao/Dropbox/notebook/data/morstar/" # make sure the last character is '/'
in_tag = "morstar"
#in_directory = "C:/Users/hsiao/Dropbox/notebook/data/test/"
#in_tag = "test"
in_parseFirstPar = True
in_window = 1

# output
out_hl_list = None
out_tag = in_tag

# outfile
out_csvfile = 'output/hl_list_'+ out_tag + '.csv'
out_featuretabfile = 'output/featurecsvfile_'+ out_tag + '.tab'
out_picklefile = 'pickle/hl_list_'+ out_tag + '.pickle'

# Example
import os

hl_list = next(os.walk(in_directory))[2] # get all filenames in the in_directory
hl_list = [os.path.join(in_directory, f) for f in hl_list] # filepathname list

hl_list = list(filter(lambda f: f.endswith(".hooklog"), hl_list)) # in case some non-hooklog file in the folder

for file in hl_list:
    hl3 = Hooklog(file, in_parseFirstPar)

# Test

# print last hl3
print("=====")
print(hl3)

# get the api call sequence of last hk3
#print("=====")
#for time, api in hl3.li:
#    print(time, api)

for i in hl3.li:
    print(i)

#print("=====")
# get the api call set of last hk3 with window size
#for api in hl3.getWinSet(in_window):
#    print(api)

# output
import pickle

out_hl_list = hl_list
out_tag = in_tag
with open(out_picklefile, 'wb') as o:
    pickle.dump(out_hl_list, o)

# output 2
with open(out_csvfile, 'wb') as o:
    o.write(b"hooklog\n")
    for h in hl_list:
        o.write(h.encode("ascii") + b"\n")

# output 3
total_api_set = set()
for file in hl_list:
    hl3 = Hooklog(file, in_parseFirstPar)
    total_api_set.update(hl3.getWinSet(in_window))

with open(out_featuretabfile, 'wb') as o:
    o.write(b"hooklog")
    for api in total_api_set:
        o.write(b"\t" + api.encode("ascii"))
    o.write(b"\n")
    
    for h in hl_list:
        hl3 = Hooklog(h, in_parseFirstPar)
        hl3_set = hl3.getWinSet(in_window)
        o.write(hl3.digitname.encode("ascii"))
        for api in total_api_set:
            if api in hl3_set:
                o.write(b"\tTrue")
            else:
                o.write(b"\tFalse")
        o.write(b"\n")

