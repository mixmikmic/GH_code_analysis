# command line tool
get_ipython().system('head -n 10 ~/MiniPythonBook/AcF701/datasets/babynames/yob2016.txt')

get_ipython().system('grep "Alice" ~/MiniPythonBook/AcF701/datasets/babynames/yob2016.txt')

import pandas as pd

baby2016 = pd.read_csv("~/MiniPythonBook/AcF701/datasets/babynames/yob2016.txt",
                           names=['name', 'gender', 'nbirth'])

baby2016.head()

# let's have a look top 5 names for girls and boys
baby2016.sort_values(['gender','nbirth'], ascending=False).groupby('gender').head(5)

sub_df = []
columns = ['name', 'gender', 'nbirth']

for year in range(1880, 2017):
    path = "~/MiniPythonBook/AcF701/datasets/babynames/yob{}.txt".format(year)
    csv  = pd.read_csv(path, names=columns)
    csv['year']=year
    
    sub_df.append(csv)
    
    df = pd.concat(sub_df, ignore_index=True)

df.info()

df.describe()

total_nbirth = df.pivot_table('nbirth', index='year', 
                              columns='gender', aggfunc=sum)

total_nbirth.head()

total_nbirth.plot(title="Total number of births, by gender and year");

