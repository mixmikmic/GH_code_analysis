import os
import json
import pandas as pd
from numpy import nan as NaN

# input
in_directory = "C:/test/VTReport/" # your VTreport dir
in_hooklog_directory = "C:/Users/hsiao/Downloads/GitHub/MotifAnalysis/hooklogs/somoto_woj/" # a hooklog dir, or None
in_tag = "somoto_woj"
in_first_seen = True
in_save_to_subdir_firstseen = True

# output
out_tag = in_tag

# outfiles
out_csvfile = 'output/VTRepo_'+ out_tag + '.csv' # original
out_wn_csvfile = 'output/VTRepo_wn_'+ out_tag + '.csv' # winnowed

# MIKE: 20170822, hack for TXT (VT report) or hooklog
run_directory = in_hooklog_directory if in_hooklog_directory != None else in_directory

# iter the directory
file_list = next(os.walk(run_directory))[2]
hash_set = set(t.split('.')[0].split("_")[0] for t in file_list)
ext = file_list[0].split('.')[-1].lower() 

print("%d files" % len(file_list))
print("%d hashes" % len(hash_set))
print(ext)
print("save to", out_csvfile)
print("save to", out_wn_csvfile)

# find all anti-virus engines and corresponding detection strings

av_set = set() # set of all anti-virus engines
csv_dict = dict()

for h in hash_set:
    # open txt file and load it as json
    with open(os.path.join(in_directory, h + '.txt')) as txt_file:    
        json_report = json.load(txt_file)
        
    # create a dictionary _dict = {engine: "detection_name"}
    _dict = dict()
    for engine in json_report['scans'].keys(): 
        scan_result = json_report['scans'].get(engine)
        if scan_result.get("detected") == True:
            result = scan_result.get("result").encode('ascii', 'ignore')
            result = result.decode("ascii").replace(',', '') # special replacement for csv
            av_set.add(engine)
            
            _dict[engine] = result
    
    # if you don't need first_seen, set in_first_seen as Fasle
    if in_first_seen:
        _dict["first_seen"] = json_report['first_seen']
    
    # attach this dictioary to csv_dict = {hash: _dict}
    csv_dict[h] = _dict

df = pd.DataFrame(csv_dict).T

# You can print the df here
df.head()
#df['AVG']

# output
df.to_csv(out_csvfile)

# MIKE: 20170731 some hacks for winnowing

# delimiter is used for spliting tokens
import re
delimiter = '\,|!|\(|\)|\[|\]|@|:|/|\.| '

# general_string to remove
general_string = ['win32','trojan','adware','generic','application','variant','downloader','not-a-virus','downware',
                 'unwanted-program','heur','troj','bundler','antifw','riskware','optional','malware','behaveslike',
                 'kcloud','agent','trojandownloader','appl','trojware','installer','trojan-downloader','virus',
                 'backdoor','injector','malware-cryptor','dropper','cryptor','bundleapp','suspicious','antifwk',
                 'adinstaller','crypt','bundleinstaller','xpack', 'hacktool','patcher','troj_gen','grayware',
                 'software','install','click','heuristic','packed','unknown','applicunwnt','dropped','trojan-clicker',
                 'net-worm','monitoringtool','worm','tool','toolbar','eldorado','autorun','hw32', 'trojan-dropper']

# short family strings that should be kept
short_family_string = ['kdz', 'ipz', 'lmn']

import string
def is_hex(s):
    return all(c in string.hexdigits for c in s)

def tk_winnow(t):
    if len(t) <= 3 and t not in short_family_string:
        return None
    elif t in general_string:
        return None
    elif is_hex(t):
        return None
    
    return t

def VT_winnow(s):
    if s is NaN: return NaN
    
    tokens = re.split(delimiter, s.lower())
    ret_tokens = list(filter(lambda x : x if x is not False else True, [tk_winnow(t) for t in tokens]))
    return ret_tokens if len(ret_tokens) != 0 else NaN

df_nw = df.copy().applymap(VT_winnow)

df_nw.head()

# output
df_nw.to_csv(out_wn_csvfile)

import shutil

#MIKE: set the interval of years 
in_years = [(0, 2000), (2001, 2010), (2011, 2014), (2015, 2017)]

def max_year(y):
    global in_years
    for min_year, max_year in in_years:
        if min_year <= y <= max_year:
            return max_year
    return -1

if in_save_to_subdir_firstseen and in_hooklog_directory and ext == "hooklog":
    first_seen_year_series = df['first_seen'].apply(lambda f: int(f.split('-')[0]))
    save_year_series = first_seen_year_series.apply(max_year)
    save_year_series.name = "save_year"
    
    for f in file_list:
        
        name = f.split('_')[0]
        save_year = save_year_series[name]
        
        new_dir = "hooklogs/" + in_tag + "_year" + str(save_year) +"/"
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
            
        shutil.copy(os.path.join(in_hooklog_directory, f), os.path.join(new_dir, f))
        
    print("new files with year is saved to", "hooklogs/" + in_tag + "_year")

