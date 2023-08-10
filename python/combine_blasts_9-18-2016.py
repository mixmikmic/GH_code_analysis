import pandas as pd
import os

wd = '/home/bay001/projects/kes_20160307/org/03_output/blast_diamond/all.blast'
# wd = '/home/bay001/projects/kes_20160307/data/diamond/' # original location
allfile = os.path.join(wd,'all.blast')

blasts = get_ipython().getoutput('find $wd*.blast -type f ! -empty')

o = open(allfile,'w')
for i in range(0,len(blasts)):
    a = open(blasts[i],'r')
    for line in a:
        o.write(line)
    a.close()
o.close()

get_ipython().system(' head /home/bay001/projects/kes_20160307/data/diamond/all.blast')



