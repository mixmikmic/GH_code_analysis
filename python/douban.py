# Import the required modules for the following codes

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
sns.set(style="white",color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15,9.27)
plt.rcParams['font.size'] = 10.0
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'

# Let's see what information does the data contain?

df = pd.read_excel('output.xls',encoding="UTF-8")
df.head()

directors = list(df['director'])
dir_dict = {x:directors.count(x) for x in set(directors)}
dir_df = pd.DataFrame.from_dict(dire,orient='index')
dir_df.columns = ['count']
dir_df.sort_values(['count'],axis=0,ascending=False,inplace=True)
dir_df['pert'] = dir_df['count']/250
dir_df[:20]

df[df.director=='克里斯托弗·诺兰']

df[df.director=='宫崎骏']

df[df.director=='史蒂文·斯皮尔伯格']

df[df.director=='王家卫']

df[df.director=='李安']

director_av_score = {x:df[df.director==x]['score'].mean() for x in set(directors)}
dire_score_df = pd.DataFrame.from_dict(director_av_score,orient='index')
dire_score_df.columns = ['av_score']
dire_score_df.sort_values(['av_score'],axis=0,ascending=False,inplace=True)
dire_score_df[:20]

year = list(df['year'])
year_dict = {x:year.count(x) for x in set(year)}
year_df = pd.DataFrame.from_dict(year_dict,orient='index')
year_df.columns = ['year_count']
new_year_df = year_df.sort_values(['year_count'],axis=0,ascending=False,inplace=False)
new_year_df['pert'] = year_df['year_count']/250
new_year_df[:20]

df[df.year==2010]

df[df.year==1994]

df[df.year==2009]

year_df.plot(kind='bar',legend=False)
plt.ylabel('number of movies in this year',size=16)
plt.title('numer of movies from year 1931 to 2016',size=20)

year_av_score = {x:df[df.year==x]['score'].mean() for x in set(year)}
year_score_df = pd.DataFrame.from_dict(year_av_score,orient='index')
year_score_df.columns = ['av_score']
year_score_df.plot(kind='bar',legend=False)
plt.ylim(8,10)
plt.ylabel('Average score of movies in the year',size=16)
plt.title('The average number of movie scores each year',size=20)

# First, let's take a look at the relationship between the average star rating of the movie and the standard deviation of the star rating

was = np.array(df['wa_star'])
wa_std = np.array(df['wa_star_std'])

sns.regplot(was,wa_std,order=2)
plt.xlabel('the weighted average of star ratings',size=16)
plt.ylabel('the standard deviation of star rating',size=16)
plt.title(' relationship between wa_star and wa_star_std',size=20)
plt.text(4.45,0.85,r'$y=-0.491x^2+3.975x-7.274,\ R^2=0.741$',size=16)

def reg(y,yname,xname,*args):
    import statsmodels.api as sm
    x = np.vstack((args)).T
    mat_x = sm.add_constant(x)
    res = sm.OLS(y,mat_x).fit()
    print(res.summary(yname=yname,xname=['cosnt']+xname))

# was_sq is the squre of the weighted average star rating

was_sq = was**2
reg(wa_std,'wa_star_std',['was','was_sq'],was,was_sq)



