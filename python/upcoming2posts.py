get_ipython().system('git pull')

import datetime
from datetime import timedelta
import os
import glob
import re



today = datetime.date.today()
yesterday = today - timedelta(1)

if yesterday.isoweekday() == 2:
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    

filename = glob.glob("_posts/" + yesterday_str + "*")[0]

with open(filename, "r") as file:
    file_text = file.read()
file_text

file_text = file_text.replace('category: upcoming', 'category: posts')
file_text = file_text.replace('category:upcoming', 'category: posts')
file_text

with open(filename, "w") as file:
    file.write(file_text)

get_ipython().system('git commit -a -m "upcoming to posts [automated]"')

# note that you have to have credentials set up to push from a notebook, otherwise you do it manually
get_ipython().system('git push')



