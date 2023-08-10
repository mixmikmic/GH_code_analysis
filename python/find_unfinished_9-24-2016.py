import pandas as pd

wd = '/home/bay001/projects/kes_20160307/data/diamond/part*.log'

d = get_ipython().getoutput("grep -L 'queries aligned.' $wd")

d

get_ipython().system(' tail /home/bay001/projects/kes_20160307/data/diamond/part_199*')

get_ipython().system(' wc -l /home/bay001/projects/kes_20160307/data/diamond/part_199.blast')



