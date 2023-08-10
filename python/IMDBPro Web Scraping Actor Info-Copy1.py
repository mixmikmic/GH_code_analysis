def launch_selenium(names_list):
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    
    import os
    import time
    import SENSITIVE as SENS
    
    # mv chrome driver from Downloads to Applications 
    chromedriver = "/Applications/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver

    url = 'https://pro-labs.imdb.com/name/' + names_list[0] + '/'
    
    driver = webdriver.Chrome(chromedriver)
    driver.get(url)
    
    loginButton = driver.find_element_by_xpath('//a[@class="log_in"]')
    loginButton.click()

    time.sleep(.5)
    loginButton = driver.find_element_by_xpath('//input[@id="auth-lwa-button"]')
    loginButton.click()

    time.sleep(.5)
    username_form = driver.find_element_by_id("ap_email")
    username_form.send_keys(SENS.username)

    password_form=driver.find_element_by_id('ap_password')
    password_form.send_keys(SENS.password)

    password_form.send_keys(Keys.RETURN)
    
    return driver


def get_actor_rankings(driver, name_list):
    #Selenium is a web browser testing automation tool
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC

    import time
    import SENSITIVE as SENS
    from IPython.display import clear_output
    import datetime
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    import pandas as pd

    import os
    import pickle
    
    
    # Load temporary pickle to store data and avoid repeats
    try:
        with open("star_scrape_progress.pkl", 'rb') as picklefile:
            df = pickle.load(picklefile)
    except:
        df = []
        
    cur_idx = 0
    for name_id in name_list:
        cur_idx += 1
        if (len(df) > 0) and (name_id in df['nconst'].values):
#             print('%s already in data.' % name_id)
            continue
        print("(%d/%d)" % (cur_idx, len(name_list)))
        url = 'https://pro-labs.imdb.com/name/' + name_id + '/'
        print("Fetching URL: " + url)
        driver.get(url+'graph/')

#         time.sleep(.5)
#         starMeterButton = driver.find_element_by_xpath('//span[text()="STARmeter"]')
#         starMeterButton.click()

#         time.sleep(.5)

        try:
            five_year = driver.find_element_by_id('five_years')
            five_year.click()
        except:
            df_tmp = pd.DataFrame([[name_id, '', '', '', 'UNAVAILABLE']], columns=['nconst', 'Name', 'Start Date', 'End Date', 'Star Ranking'])
            if len(df) == 0:
                df = df_tmp
            else:
                df = df.append(df_tmp)
            continue

        graph_div = driver.find_element_by_id('ranking_graph')

        location = graph_div.find_elements_by_tag_name('rect')[1:]

        name = driver.find_elements_by_class_name('display-name')[1].find_element_by_tag_name('a').text

        star_meter_data = []
        for i in range(1, len(location)+1):
    #         clear_output(wait=True)
#             print(i)
            loc = graph_div.find_elements_by_tag_name('rect')[i]
            driver.find_element_by_class_name('current_rank').find_element_by_tag_name('span').click()
        #     time.sleep(.1)
            try:
                loc.click()
            except:
                time.sleep(0.5)
            g = graph_div.find_elements_by_tag_name('tspan')[-2:]
        #     print("%s: \n\t%s" % (g[0].text, g[1].text))
        #     print("x = " + loc.get_attribute('x'))
            dates = g[0].text.split('-')
            start_date = datetime.datetime.strptime(dates[0].strip(), '%b %d, %Y')
            end_date = datetime.datetime.strptime(dates[1].strip(), '%b %d, %Y')
            star_meter = int(g[1].text.split(':')[-1].strip().replace(',',''))
            star_meter_data.append([i, name_id, name, start_date, end_date, star_meter])


        df_tmp = pd.DataFrame([i[1:] for i in star_meter_data], index=[i[0] for i in star_meter_data], columns=['nconst', 'Name', 'Start Date', 'End Date', 'Star Ranking'])
        if len(df) == 0:
            df = df_tmp
        else:
            df = df.append(df_tmp)
        
        with open('star_scrape_progress.pkl', 'wb') as picklefile:
            pickle.dump(df, picklefile)
        
        clear_output(wait=True)
    
    driver.quit()
    return df

# # TESTING
# names = []
# names.append('nm0425005')
# names.append('nm0000129')
# names.append('nm0000230')
# names.append('nm3915784')
# names.append('nm9152705')

# driver = launch_selenium(names)
# df = get_actor_rankings(driver, names)

# # for u in url[1:]:
# #     df = df.append(get_actor_ranking(u))

# df.tail(10)





# import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline

# fig, ax = plt.subplots(1,1, figsize=(12,8))
# all_names = []
# for name in pd.unique(df['Name']):
#     df[df['Name'] == name].plot('Start Date', 'Star Ranking', ax=ax)
#     all_names.append(name)
# ax.legend(all_names)





# Work with and filter basic movie title info
import pandas as pd
import numpy as np
title_basics_df = pd.read_csv('Data/title.basics.tsv', sep='\t')

def clean_year(y):
    import numpy as np
    try:
        return int(y)
    except:
        return np.nan

def clean_genre(y):
    y = str(y)
    if y == '\\N':
        return ''
    return y.split(',')[0].strip()

import datetime
import numpy as np
print(len(title_basics_df))
title_basics_df.drop('endYear', axis=1, inplace=True)
title_basics_df['startYear'] = title_basics_df['startYear'].apply(clean_year)
title_basics_df['runtimeMinutes'] = title_basics_df['runtimeMinutes'].apply(clean_year)
title_basics_df['genres'] = title_basics_df['genres'].apply(clean_genre)
title_basics_df.dropna(inplace=True, how='any', subset=['startYear', 'runtimeMinutes'])
print(len(title_basics_df))

title_basics_df.head(20)

mask = ((title_basics_df['startYear'] >= 2014) &
        (title_basics_df['startYear'] <= 2017) &
        (title_basics_df['titleType'] == 'movie') &
        (title_basics_df['isAdult'] == 0) & 
        (title_basics_df['runtimeMinutes'] > 80) &
        (title_basics_df['genres'] != '') &
        (title_basics_df['genres'] != 'Documentary'))

title_cast_df = pd.read_csv('Data/title.principals.tsv', sep='\t')
title_ratings_df = pd.read_csv('Data/title.ratings.tsv', sep='\t')

title_cast_df.head()

title_ratings_df.head()

titles = title_basics_df[mask].merge(title_cast_df, on='tconst')
titles = titles.merge(title_ratings_df, on='tconst')
titles['leadActor'] = titles['principalCast'].apply(lambda x: x.split(',')[0])

len(titles)

import pickle
with open('my_data.pkl', 'wb') as picklefile:
   pickle.dump(titles['tconst'].values, picklefile)

all_names = set()
for cast_list in titles['leadActor']:
    for name in [cast_list]:
        all_names.add(name)

print("Number of movies: %d" % len(titles))
print("Number of unique actors: %d" % len(all_names))

len(all_names)*12/3600



# Load IMDB title scrapy results
# Read json file
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

with open('imdb_spider/import_20Jan18_1.json', 'r') as f:
    my_data = json.load(f)

imdb_info = pd.DataFrame(my_data)

imdb_mask = ((imdb_info['budget'] != '') &
             (imdb_info['metacritic_score'] != '') &
             (imdb_info['opening'] != '') &
             (~imdb_info['mpaa_rating'].isin(['', 'UNRATED', 'NOT RATED', 'TV-14'])))

imdb_info = imdb_info[imdb_mask]
imdb_info.filter('mpaa_rating NOT IN ["UNRATED", "NOT RATED", "TV-14]')

len(imdb_info)

imdb_info['budget'] = imdb_info['budget'].apply(int)
imdb_info['budget_mil'] = imdb_info['budget']/1000000.
imdb_info['opening'] = imdb_info['opening'].apply(int)
imdb_info['opening_mil'] = imdb_info['opening']/1000000.
imdb_info['metacritic_score'] = imdb_info['metacritic_score'].apply(int)
imdb_info['release'] = pd.to_datetime(imdb_info['release'].apply(lambda x: x.split('(')[0].strip()))
imdb_info['tconst'] = imdb_info['title_id']
imdb_info.drop('title_id', inplace=True, axis=1)
imdb_info.head()

# Merge IMDB Info with titles data
titles_all = imdb_info.merge(titles, on='tconst')

# Confirm:
print('Length of IMDB Info: %d' % len(imdb_info))
print('Length of Titles DataFrame: %d' % len(titles))
print('Length of Final Merged DataFrame: %d' % len(titles_all))

titles_all.head()

col_names = ['tconst', 'principalCast']
expanded_data = []
for idx, row in titles_all[col_names].iterrows():
    for name in row['principalCast'].split(','):
        expanded_data.append([row['tconst'], name.strip()])
expanded_data = pd.DataFrame(expanded_data, columns=['tconst', 'nconst'])
expanded_data.head()

# Load names database and merge with required
import csv
names_data = []
with open("Data/name.basics.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        names_data.append(row)

column_names = names_data[0]

df_names = pd.DataFrame(data=names_data[1:], columns=column_names)

df_names_movie_link = expanded_data.merge(df_names, how='left', on='nconst')
df_names_movie_link.head()

# Clean Up (limit to single primary profession)
df_names_movie_link['primaryProfession'] = df_names_movie_link['primaryProfession'].apply(lambda x: x.split(',')[0].strip())
df_names_movie_link.head()

# merge this with titles_all dataframe
df_total = df_names_movie_link.merge(titles_all, how='left', on='tconst')
df_total.head()

# Number of unique actors
num_actors = len(pd.unique(df_total['nconst']))
print('Number of unique actors: %d' % num_actors)
print('Estimated Selenium Scraping Time: %0.2f Hours' % (num_actors*12/3600))



## RUN SELENIUM IMPORT
names = list(df_total['nconst'].values)
driver = launch_selenium(names)
df = get_actor_rankings(driver, names)





# import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline

# fig, ax = plt.subplots(1,1, figsize=(12,8))
# all_names = []
# for name in pd.unique(df['Name']):
#     df[df['Name'] == name].plot('Start Date', 'Star Ranking', ax=ax)
#     all_names.append(name)
# ax.legend(all_names)

import pickle
with open("my_data.pkl", 'rb') as picklefile:
    links = list(pickle.load(picklefile))
len(links)



from bs4 import BeautifulSoup
import requests

# r = requests.get('http://www.imdb.com/title/tt0137204/')
r = requests.get('http://www.imdb.com/title/tt0340855/')
soup  = BeautifulSoup(r.text)

release_date = soup.find('meta', {'itemprop': 'datePublished'}).get_attribute_list('content')[0]
print(release_date)

director = soup.find_all('span', {'itemprop': 'creator'})[0].text.replace('\n','')
studio = soup.find_all('span', {'itemprop': 'creator'})[1].text.replace('\n','')
print(director)
print(studio)

for elem in soup.find_all('h3', {'class': 'subheading'}):
    print(elem.text)





# Read json file
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

with open('imdb_spider/test.json', 'r') as f:
    my_data = json.load(f)

imdb_info = pd.DataFrame(my_data)

imdb_mask = ((imdb_info['budget'] != '') &
             (imdb_info['metacritic_score'] != '') &
             (imdb_info['opening'] != '') &
             (~imdb_info['mpaa_rating'].isin(['', 'UNRATED', 'NOT RATED', 'TV-14'])))

imdb_info = imdb_info[imdb_mask]
imdb_info.filter('mpaa_rating NOT IN ["UNRATED", "NOT RATED", "TV-14]')

len(imdb_info)

imdb_info['budget'] = imdb_info['budget'].apply(int)
imdb_info['budget_mil'] = imdb_info['budget']/1000000.
imdb_info['opening'] = imdb_info['opening'].apply(int)
imdb_info['opening_mil'] = imdb_info['opening']/1000000.
imdb_info['metacritic_score'] = imdb_info['metacritic_score'].apply(int)
imdb_info['release'] = pd.to_datetime(imdb_info['release'].apply(lambda x: x.split('(')[0].strip()))

s = sns.lmplot(x='budget_mil', y='opening_mil', 
           data=imdb_info, 
           hue='mpaa_rating', 
           fit_reg=False, 
           size=5, 
           aspect=2, 
           scatter_kws={'alpha': 0.5, 
                        's': 50},
           legend=False);
s.ax.plot([0, 300], [0, 300], 'k--', lw=3, label='Opening Weekend Profit');
s.ax.set_xlim(0, 50)
s.ax.set_ylim(0, 50)
s.ax.legend();



imdb_info.plot('budget', 'opening', kind='scatter')





import csv
names_data = []
with open("Data/name.basics.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        names_data.append(row)

column_names = names_data[0]

df = pd.DataFrame(data=names_data[1:], columns=column_names)

df.head(20)



