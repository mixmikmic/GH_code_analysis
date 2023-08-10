import time, random, pandas as pd, pytz
from dateutil import parser as date_parser
from datetime import datetime as dt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# define the options for launching chrome
chrome_options = Options()
chrome_options.add_argument('--disable-extensions')
chrome_options.binary_location = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
executable_path = 'chromedriver.exe'
maximize_window = False

# define the pause durations
short_pause_min = 1
short_pause_max = 2

# define pause functions
def pause(duration_min, duration_max):
    duration = (duration_max - duration_min) * random.random() + duration_min
    time.sleep(duration)

def pause_short():
    pause(short_pause_min, short_pause_max)

# determine the time the script started at
start_time = time.time()
print('start time {}'.format(dt.fromtimestamp(start_time).strftime('%H:%M:%S')))

# load the check-ins data set (which currently lacks full details)
df = pd.read_csv('data/untappd.csv')

# create the lists of urls to visit for beers, breweries, and venues
# for efficiency, these lists are unique so we only visit each page once
beer_urls = df[df['beer_url'].notnull()]['beer_url'].unique()
brewery_urls = df[df['brewery_url'].notnull()]['brewery_url'].unique()
venue_urls = df[df['venue_url'].notnull()]['venue_url'].unique()

# launch the chrome driver, then clear cookies and cache
driver = webdriver.Chrome(executable_path=executable_path, chrome_options=chrome_options)
driver.delete_all_cookies()
if maximize_window:
    driver.maximize_window()

# extracts/returns all the details (style, abv, ibu, total check-ins, average rating) from a beer's page
def get_beer_details():
    
    beer_style_query = '//div[@class="name"]/p[@class="style"]'
    try:
        beer_style_item = driver.find_elements(By.XPATH, beer_style_query)[0]
        beer_style = beer_style_item.text
    except:
        beer_style = None
    
    beer_abv_query = '//div[@class="details"]/p[@class="abv"]'
    try:
        beer_abv_item = driver.find_elements(By.XPATH, beer_abv_query)[0]
        beer_abv = beer_abv_item.text
    except:
        beer_abv = None
    
    beer_ibu_query = '//div[@class="details"]/p[@class="ibu"]'
    try:
        beer_ibu_item = driver.find_elements(By.XPATH, beer_ibu_query)[0]
        beer_ibu = beer_ibu_item.text
    except:
        beer_ibu = None
    
    beer_total_checkins_query = '//div[@class="stats"]/p/span[@class="count"]'
    try:
        beer_total_checkins_item = driver.find_elements(By.XPATH, beer_total_checkins_query)[0]
        beer_total_checkins = beer_total_checkins_item.text
    except:
        beer_total_checkins = None
    
    beer_avg_rating_query = '//div[@class="details"]/p[@class="rating"]'
    try:
        beer_avg_rating_item = driver.find_elements(By.XPATH, beer_avg_rating_query)[0]
        beer_avg_rating = beer_avg_rating_item.text
    except:
        beer_avg_rating = None
    
    return beer_style, beer_total_checkins, beer_avg_rating, beer_abv, beer_ibu

# get all the beers' details and save to a dict with the URLs as keys
beer_details = {}
for beer_url in beer_urls:
    driver.get(beer_url)
    beer_style, beer_total_checkins, beer_avg_rating, beer_abv, beer_ibu = get_beer_details()
    beer_details[beer_url] = {'beer_style' : beer_style,
                              'beer_total_checkins' : beer_total_checkins,
                              'beer_avg_rating' : beer_avg_rating,
                              'beer_abv' : beer_abv,
                              'beer_ibu' : beer_ibu}    
    pause_short()

# extracts/returns all the details (place, type, avg rating, total check-ins) from a brewery's page
def get_brewery_details():
    
    brewery_place_query = '//div[@class="basic"]/div[@class="name"]/p[@class="brewery"]'
    try:
        brewery_place_item = driver.find_elements(By.XPATH, brewery_place_query)[0]
        brewery_place = brewery_place_item.text
    except:
        brewery_place = None
    
    brewery_type_query = '//div[@class="name"]/p[@class="style"]'
    try:
        brewery_type_item = driver.find_elements(By.XPATH, brewery_type_query)[0]
        brewery_type = brewery_type_item.text
    except:
        brewery_type = None
    
    brewery_avg_rating_query = '//div[@class="content"]/div/p[@class="rating"]/span[@class="num"]'
    try:
        brewery_avg_rating_item = driver.find_elements(By.XPATH, brewery_avg_rating_query)[0]
        brewery_avg_rating = brewery_avg_rating_item.text
    except:
        brewery_avg_rating = None
    
    brewery_total_checkins_query = '//div[@class="stats"]/p/span[@class="count"]'
    try:
        brewery_total_checkins_item = driver.find_elements(By.XPATH, brewery_total_checkins_query)[0]
        brewery_total_checkins = brewery_total_checkins_item.text
    except:
        brewery_total_checkins = None
    
    return brewery_place, brewery_type, brewery_total_checkins, brewery_avg_rating

# get all the breweries' details and save to a dict with the URLs as keys
brewery_details = {}
for brewery_url in brewery_urls:
    if pd.notnull(brewery_url):
        driver.get(brewery_url)
        brewery_place, brewery_type, brewery_total_checkins, brewery_avg_rating = get_brewery_details()
        brewery_details[brewery_url] = {'brewery_place' : brewery_place,
                                        'brewery_type' : brewery_type,
                                        'brewery_total_checkins' : brewery_total_checkins,
                                        'brewery_avg_rating' : brewery_avg_rating}
        pause_short()

# extracts/returns all the details (place, type, total check-ins, lat/long) from a venue's page
def get_venue_details():
    
    venue_place_query = '//div[@class="header-meta"]/p'
    try:
        venue_place_item = driver.find_elements(By.XPATH, venue_place_query)[1]
        venue_place = venue_place_item.text
    except:
        venue_place = None
    
    venue_type_query = '//div[@class="venue-name"]/h2'
    try:
        venue_type_item = driver.find_elements(By.XPATH, venue_type_query)[0]
        venue_type = venue_type_item.text
    except:
        venue_type = None
    
    venue_total_checkins_query = '//div[@class="stats"]/ul/li'
    try:
        venue_total_checkins_item = driver.find_elements(By.XPATH, venue_total_checkins_query)[0]
        venue_total_checkins = venue_total_checkins_item.text
    except:
        venue_total_checkins = None
    
    venue_lat_query = '//meta[@property="place:location:latitude"]'
    try:
        venue_lat_item = driver.find_elements(By.XPATH, venue_lat_query)[0]
        venue_lat = venue_lat_item.get_attribute('content')
    except:
        venue_lat = None
    
    venue_lon_query = '//meta[@property="place:location:longitude"]'
    try:
        venue_lon_item = driver.find_elements(By.XPATH, venue_lon_query)[0]
        venue_lon = venue_lon_item.get_attribute('content')
    except:
        venue_lon = None
    
    return venue_place, venue_type, venue_total_checkins, venue_lat, venue_lon

# get all the venues' details and save to a dict with the URLs as keys
venue_details = {}
for venue_url in venue_urls:
    driver.get(venue_url)
    venue_place, venue_type, venue_total_checkins, venue_lat, venue_lon = get_venue_details()
    venue_details[venue_url] = {'venue_place' : venue_place,
                                'venue_type' : venue_type,
                                'venue_total_checkins' : venue_total_checkins,
                                'venue_lat' : venue_lat,
                                'venue_lon' : venue_lon}    
    pause_short()

# all done, close the webdriver
driver.close()

# calculate the current time and the elapsed time
current_time = time.time()
print('current time {}'.format(dt.fromtimestamp(current_time).strftime('%H:%M:%S')))
print('elapsed time: {:,.1f} secs'.format(current_time-start_time))

# first, create new columns in the dataframe to contain the details
df['beer_style'] = None
df['beer_total_checkins'] = None
df['beer_avg_rating'] = None
df['beer_abv'] = None
df['beer_ibu'] = None
df['brewery_place'] = None
df['brewery_type'] = None
df['brewery_total_checkins'] = None
df['brewery_avg_rating'] = None
df['venue_place'] = None
df['venue_type'] = None
df['venue_total_checkins'] = None
df['venue_lat'] = None
df['venue_lon'] = None

# for each url in each list of beer urls, get all the dataframe rows that contain this url
# then find this url in the dict and copy each detail value from dict to the corresponding column in the df
for beer_url in beer_urls:
    labels = df[df['beer_url']==beer_url].index
    df.loc[labels, 'beer_style'] = beer_details[beer_url]['beer_style']
    df.loc[labels, 'beer_total_checkins'] = beer_details[beer_url]['beer_total_checkins']
    df.loc[labels, 'beer_avg_rating'] = beer_details[beer_url]['beer_avg_rating']
    df.loc[labels, 'beer_abv'] = beer_details[beer_url]['beer_abv']
    df.loc[labels, 'beer_ibu'] = beer_details[beer_url]['beer_ibu']

# do the same for the brewery urls
for brewery_url in brewery_urls:
    labels = df[df['brewery_url']==brewery_url].index
    df.loc[labels, 'brewery_place'] = brewery_details[brewery_url]['brewery_place']
    df.loc[labels, 'brewery_type'] = brewery_details[brewery_url]['brewery_type']
    df.loc[labels, 'brewery_total_checkins'] = brewery_details[brewery_url]['brewery_total_checkins']
    df.loc[labels, 'brewery_avg_rating'] = brewery_details[brewery_url]['brewery_avg_rating']

# do the same for the venue urls
for venue_url in venue_urls:
    labels = df[df['venue_url']==venue_url].index
    df.loc[labels, 'venue_place'] = venue_details[venue_url]['venue_place']
    df.loc[labels, 'venue_type'] = venue_details[venue_url]['venue_type']
    df.loc[labels, 'venue_total_checkins'] = venue_details[venue_url]['venue_total_checkins']
    df.loc[labels, 'venue_lat'] = venue_details[venue_url]['venue_lat']
    df.loc[labels, 'venue_lon'] = venue_details[venue_url]['venue_lon']

# define parse functions to clean strings and change datatypes for each field necessary
def parse_avg_rating(val):
    try:
        return float(val.strip('(').strip(')'))
    except:
        return None
    
def parse_beer_brewery_checkins_count(val):
    try:
        if 'M+' in val:
            return float(val.replace(',', '').strip('M+')) * 1000000
        else:
            return float(val.replace(',', ''))
    except:
        return None
     
def parse_venue_checkins_count(val):
    try:
        return float(val.replace(',', '').strip('\nTOTAL'))
    except:
        return None
    
def parse_beer_ibu(val):
    try:
        return float(val.strip(' IBU'))
    except:
        return None
    
def parse_beer_abv(val):
    try:
        return float(val.strip('% ABV'))
    except:
        return None

# clean/transform each column using my parse functions or pandas' built-in strip/astype methods
df['beer_total_checkins'] = df['beer_total_checkins'].map(parse_beer_brewery_checkins_count)
df['beer_avg_rating'] = df['beer_avg_rating'].map(parse_avg_rating)
df['beer_ibu'] = df['beer_ibu'].map(parse_beer_ibu)
df['beer_abv'] = df['beer_abv'].map(parse_beer_abv)
df['brewery_avg_rating'] = df['brewery_avg_rating'].map(parse_avg_rating)
df['brewery_total_checkins'] = df['brewery_total_checkins'].map(parse_beer_brewery_checkins_count)
df['venue_total_checkins'] = df['venue_total_checkins'].map(parse_venue_checkins_count)
df['venue_place'] = df['venue_place'].str.strip(' (Map)')
df['venue_lat'] = df['venue_lat'].astype(float)
df['venue_lon'] = df['venue_lon'].astype(float)

# sort the column names to be a bit more intuitive
cols = ['date_pacific_tz', 'beer_name', 'beer_style', 'brewery_name', 'brewery_place', 'brewery_type',
        'rating', 'beer_avg_rating', 'brewery_avg_rating', 'beer_abv', 'beer_ibu', 'beer_total_checkins',
        'brewery_total_checkins', 'venue_name', 'venue_type', 'venue_place', 'venue_lat', 'venue_lon',
        'venue_total_checkins', 'checkin_id', 'beer_url', 'brewery_url', 'venue_url']
df = df.reindex(columns=cols)

# show a slice of the final dataframe
df.head()

# save to csv
df.to_csv('data/untappd_details.csv', index=False, encoding='utf-8')

# calculate the end time and the elapsed time
end_time = time.time()
print('end time {}'.format(dt.fromtimestamp(end_time).strftime('%H:%M:%S')))
print('elapsed time: {:,.1f} secs'.format(end_time-start_time))



