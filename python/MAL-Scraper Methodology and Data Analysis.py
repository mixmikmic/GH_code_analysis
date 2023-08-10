from requests import get
from bs4 import BeautifulSoup
import time
import random
import math
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)

url = 'https://myanimelist.net/reviews.php?t=anime&p=6'
headers = {
        "User-Agent": "mal review scraper for research."
}
response = get(url)
print(response.status_code)

html_soup = BeautifulSoup(response.text, 'html.parser')

review_containers = html_soup.find_all('div', class_ = 'borderDark')
len(review_containers)

review = review_containers[22]

review_id = review.find_all(
    'div', attrs={'style':"float: left; display: none; margin: 0 10px 10px 0"})[0].attrs['id']
review_id

review_id = review_id.replace('score', '')
review_id

anime_id = review.div.find('a', class_='hoverinfo_trigger').attrs['rel'][0]
anime_id

anime_id = anime_id.replace('#revInfo', '')
anime_id

review_element = review.div
anime_name = review_element.find('a', class_='hoverinfo_trigger').text         
username = review_element.find_all('td')[1].a.text
review_date = review_element.div.div.text
episodes_seen = review_element.div.find_all('div')[1].text.strip().split(' ')[0]

overall_rating = review_element.div.find_all('div')[2].text.strip().split('\n')[1]
story_rating = review.find_all('td', class_='borderClass')[3].text
animation_rating = review.find_all('td', class_='borderClass')[5].text
sound_rating = review.find_all('td', class_='borderClass')[7].text
character_rating = review.find_all('td', class_='borderClass')[9].text    
enjoyment_rating = review.find_all('td', class_='borderClass')[11].text

#Review helpful counts
helpful_counts = review_element.find('span').text

#Review Body
body1 = review.select('div.spaceit.textReadability.word-break.pt8')[0].contents[4].strip()
body2 = review.select('div.spaceit.textReadability.word-break.pt8')[0].contents[5].text.strip()
review_body = (body1 + ' ' + body2)
review_body[0:500]

review_body = (body1 + ' ' + body2).replace('\n', ' ').replace('\r', ' ')
review_body[0:500]

def run_query(DB, q):
    with sqlite3.connect(DB) as conn:
        return pd.read_sql(q,conn)

def run_command(DB, c):
    with sqlite3.connect(DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        conn.isolation_level = None
        conn.execute(c)
        
def run_inserts(DB, c, values):
    with sqlite3.connect(DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        conn.isolation_level = None
        conn.execute(c, values) 

DB = 'anime.db'

insert_command = '''
INSERT OR IGNORE INTO reviews(
    review_id,
    anime_id, 
    username, 
    review_date,
    episodes_seen,
    overall_rating,
    story_rating,
    animation_rating,
    sound_rating,
    character_rating,
    enjoyment_rating,
    helpful_counts,    
    review_body
    ) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''

run_inserts(
    DB, insert_command,(
        int(review_id), int(anime_id), username, review_date, \
        episodes_seen, int(overall_rating), \
        int(story_rating), int(animation_rating), int(sound_rating), \
        int(character_rating), int(enjoyment_rating), int(helpful_counts), review_body
    )
)

sleep_min = 2
sleep_max = 5

start_time = time.time()
for i in range(1, 4):
    time.sleep(random.uniform(sleep_min, sleep_max))
    print('Loop #: {}'.format(i))
    print('Elapsed Time: {} seconds'.format(time.time() - start_time))
      

sleep_min = 2
sleep_max = 5

log = []
for i in range(30000, 30003):
    time.sleep(random.uniform(sleep_min, sleep_max))
    url = 'https://myanimelist.net/reviews.php?t=anime&p={}'.format(i)
    headers = {
        "User-Agent": "mal review scraper for research."
    }

    print('Scraping: {}'.format(url))

    
    try:
        response = get(url, headers=headers, timeout = 10)
    except:
        print('Request timeout')
        log.append('Request timeout for {}'.format(url))
        pass

    if response.status_code != 200:
        print('Status code: {}'.format(response.status_code))
        log.append('Status code: {0} {1}'.format(response.status_code, url))
        pass

print(log)

start_time = time.time()
sleep_min = 2
sleep_max = 5
page_start = 1
page_end = 5

for i in range(page_start, (page_end+1)):
    time.sleep(random.uniform(sleep_min, sleep_max))
    current_time = time.time()
    elapsed_time = current_time - start_time
    requests = i    

    print('Requests Completed: {}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))
    print('Elapased Time: {} minutes'.format(elapsed_time/60))
    if requests == page_end:
        print('Scrape Complete')
        break
    print('Pausing...')    
       

testq = '''
SELECT 
    t.tag_id tag_id,
    t.tag_name Genre,
    a.anime_id,
    a.overall_rating
FROM animes a
INNER JOIN anime_tags at ON a.anime_id = at.anime_id
INNER JOIN tags t ON at.tag_id = t.tag_id

'''

table1 = run_query(DB, testq)
table1.tail()

#Handle null values
table1['overall_rating_std'] = pd.to_numeric(table1['overall_rating'], errors='coerce').dropna()
table1['overall_rating_mean'] = pd.to_numeric(table1['overall_rating'], errors='coerce').dropna()
table1['sample_size'] = 0

table1_cleaned = table1.groupby(['tag_id', 'Genre'], as_index = False).agg({'overall_rating_std': np.std, 'overall_rating_mean': np.mean, 'sample_size': len})
table1_cleaned = table1_cleaned.sort_values(by=['overall_rating_mean'], ascending = False).head(10)
table1_cleaned

y = table1_cleaned['Genre']
x = table1_cleaned['overall_rating_mean']
std = table1_cleaned['overall_rating_std']
xerr = 1.96*std/np.sqrt(table1_cleaned['sample_size']).values

fig = plt.figure(figsize=(10,5))
ax = sns.barplot(x, y, xerr=xerr, palette='GnBu_d')
ax.set_xlabel('Average Rating')
plt.show()

q2 = '''
WITH anime_genre_counts AS
    (SELECT
        a.anime_id,
        COUNT(at.anime_id) Number_of_genres,
        AVG(a.overall_rating) overall_rating
    FROM animes a
    INNER JOIN anime_tags at ON a.anime_id = at.anime_id
    INNER JOIN tags t ON at.tag_id = t.tag_id
    GROUP BY 1
    )

SELECT
    Number_of_genres,
    COUNT(anime_id) sample_size,
    AVG(overall_rating) Average_rating
FROM anime_genre_counts
GROUP BY 1
ORDER BY 3 DESC
'''

table2 = run_query(DB, q2)
table2

q3 = '''
WITH anime_studios AS
    (SELECT
        s.studio_name,
        COUNT(a.anime_id) animes_produced,
        AVG(a.overall_rating) overall_rating,
        episodes_total
    FROM animes a
    INNER JOIN studios s ON s.studio_id = a.studio_id
    WHERE episodes_total > 10
    GROUP BY 1
    )

SELECT
    studio_name,
    animes_produced,
    overall_rating average_ratings
FROM anime_studios
WHERE animes_produced > 10
ORDER BY average_ratings DESC
LIMIT 10
'''

table3 = run_query(DB, q3)
table3

q4 = '''
SELECT
    source_material,
    COUNT(anime_id) sample_size,
    AVG(overall_rating) average_ratings
FROM animes
GROUP BY source_material
ORDER BY average_ratings DESC
LIMIT 10
'''

table4 = run_query(DB, q4)
table4

y = table4['source_material']
x = table4['average_ratings']

fig = plt.figure(figsize=(10,5))
ax = sns.barplot(x, y, palette='GnBu_d')
ax.set_xlabel('Average Rating')
ax.set_ylabel('Source Material')
plt.show()



