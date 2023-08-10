from github import Github
import pandas as pd
import numpy as np
import getpass

def h_fmt(num, suffix='B'):
    """ Convert bytes to human readable format """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

username = input("Username: ")  # rc-softdev-admin
print("Enter Password for Ghub user {0}".format(username))
g = Github(username, getpass.getpass())

repo_info = [[r.name, h_fmt(r.size * 1000), r.size * 1000] for r in g.get_user().get_repos()]
df = pd.DataFrame(data=repo_info, columns=['repo','size -h','kb'])
print("Number of repos: {0} \nTotal size: ~{1}".format(len(df), h_fmt(df['kb'].values.sum())))

# show top 10
df.sort_values('kb', ascending=False).head(n=10)

import numpy as np
import scipy.special
from bokeh.plotting import figure, show, output_notebook, vplot
output_notebook()

p1 = figure(title="Repo size")
hist, edges = np.histogram(df['kb'], density=False, bins=100)
p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
p1.xaxis.axis_label = 'Size (b)'
p1.yaxis.axis_label = 'Number'
show(p1)

