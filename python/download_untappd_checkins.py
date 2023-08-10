import time, random, pandas as pd, pytz
from dateutil import parser as date_parser
from datetime import datetime as dt
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

from keys import untappd_username, untappd_password

# only get n check-ins, or get all of them if 0
limit_checkin_count = 0

# define the url to log into untappd and the url for the user's profile
login_url = 'https://untappd.com/login'
profile_url = 'https://untappd.com/user/{}'.format(untappd_username)

# define html element ids for username and password input boxes
username_box_id = 'username'
password_box_id = 'password'

# define xpath queries to find the html elements of interest
show_more_button_query = '//a[@class="yellow button more_checkins more_checkins_logged track-click"]'
checkin_item_query = '//div[@id="main-stream"]/div[@class="item"]'
beer_name_query = '//div[@id="{}"]/div[@class="checkin"]/div[@class="top"]/p[@class="text"]/a'
count_query = '//div[@class="stats-bar"]/div[@class="stats"]/a[@href="/user/{}"]/span[@class="stat"]'
rating_query = '//div[@id="{}"]/div[@class="checkin"]/div[@class="top"]/p[@class="checkin-comment"]/span[contains(@class, "rating")]'
date_query = '//div[@id="{}"]/div[@class="checkin"]/div[@class="feedback"]/div[@class="bottom"]/a[@class="time timezoner track-click"]'

# define the pause durations
short_pause_min = 1
short_pause_max = 2
medium_pause_min = 3
medium_pause_max = 4
long_pause_min = 5
long_pause_max = 6

# define the options for launching chrome
chrome_options = Options()
chrome_options.add_argument('--disable-extensions')
chrome_options.binary_location = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
executable_path = 'chromedriver.exe'
maximize_window = False

# define pause functions
def pause(duration_min, duration_max):
    duration = (duration_max - duration_min) * random.random() + duration_min
    time.sleep(duration)

def pause_short():
    pause(short_pause_min, short_pause_max)

def pause_medium():
    pause(medium_pause_min, medium_pause_max)
    
def pause_long():
    pause(long_pause_min, long_pause_max)

def get_details(names_urls):

    # initialize the name and link variables with empty strings
    beer_name = ''
    beer_url = ''
    brewery_name = ''
    brewery_url = ''
    venue_name = ''
    venue_url = ''

    # for each name+link pair we found, see if it's a beer, a brewery, or a venue
    for name, url in names_urls:
        if '/b/' in url:
            beer_name = name
            beer_url = url
        elif '/w/' in url:
            brewery_name = name
            brewery_url = url
        elif '/v/' in url:
            venue_name = name
            venue_url = url

    return beer_name, beer_url, brewery_name, brewery_url, venue_name, venue_url

# determine the time the script started at
start_time = time.time()
print('start time {}'.format(dt.fromtimestamp(start_time).strftime('%H:%M:%S')))

# launch the chrome driver, then clear cookies and cache
driver = webdriver.Chrome(executable_path=executable_path, chrome_options=chrome_options)
driver.delete_all_cookies()
if maximize_window:
    driver.maximize_window()

# log into untappd
driver.get(login_url)
pause_short()

username_box = driver.find_element_by_id(username_box_id)
username_box.clear()
username_box.send_keys(untappd_username)
username_box.send_keys(Keys.TAB)
pause_short()

password_box = driver.find_element_by_id(password_box_id)
password_box.clear()
password_box.send_keys(untappd_password)
pause_short()

password_box.send_keys(Keys.ENTER)
pause_medium()

#close the app download ad banner if it's up
try:
    driver.switch_to.frame(driver.find_element_by_tag_name('iframe'))
    driver.find_elements(By.XPATH, '//div[@id="branch-banner-close"]')[0].click()
    driver.switch_to.default_content()
    pause_short()
except:
    pass

# go to the user's profile page
driver.get(profile_url)
pause_short()

# get the count of total check-ins
pause_medium()
checkin_count_item = driver.find_elements(By.XPATH, count_query.format(untappd_username))[0]
checkin_count = int(checkin_count_item.text.replace(',', ''))
print('{:,}'.format(checkin_count))

count_found = 0
scroll_count = 0
checkin_count = limit_checkin_count if limit_checkin_count > 0 else checkin_count

# scroll to the bottom of the page
actions = ActionChains(driver)
actions.key_down(Keys.END).key_up(Keys.END).perform()
pause_short()

# until you've found all the check-ins you expect, click 'show more' button, scroll down, repeat
while count_found < checkin_count:

    # click the 'show more' button then pause while the new page data loads
    driver.find_elements(By.XPATH, show_more_button_query)[0].click()
    pause_long()

    # tab off the 'show more' button then hit the end key
    actions.key_down(Keys.SHIFT).key_down(Keys.TAB)                    .key_up(Keys.TAB).key_up(Keys.SHIFT)                        .key_down(Keys.END).key_up(Keys.END).perform()
    pause_short()
    
    # increment the counter and count how many check-in items are on the page now
    scroll_count += 1
    count_found = len(driver.find_elements(By.XPATH, checkin_item_query))
    print('scroll count: {}, found: {:,} check-ins total'.format(scroll_count, count_found))

# report how many total check-in items were found in the end, and the current time
checkin_items = driver.find_elements(By.XPATH, checkin_item_query)[:checkin_count]
current_time = time.time()
print('found {:,} check-ins'.format(len(checkin_items)))
print('current time {}'.format(dt.fromtimestamp(current_time).strftime('%H:%M:%S')))
print('elapsed time so far: {:,.1f} secs'.format(current_time-start_time))

# loop through each check-in item and get the beer, brewery, and venue details
checkins = []
for checkin_item in checkin_items:
    
    # get the check-in id then the names and links for the beer, brewery, and venue
    checkin_item_id = checkin_item.get_attribute('id')
    text_items = driver.find_elements(By.XPATH, beer_name_query.format(checkin_item_id))
    names_urls = [(item.text, item.get_attribute('href')) for item in text_items]
    
    # get the beer, brewery, and venue details
    beer_name, beer_url, brewery_name, brewery_url, venue_name, venue_url = get_details(names_urls)
    
    # when we're getting those details, get the rating
    try:
        rating_item = driver.find_elements(By.XPATH, rating_query.format(checkin_item_id))[0]
        rating = int(rating_item.get_attribute('class').split(' r')[1]) / 100.
    except:
        rating = None
    
    # then get the date
    date_item = driver.find_elements(By.XPATH, date_query.format(checkin_item_id))[0]
    date = date_item.get_attribute('data-gregtime')
    
    # to get the style, public rating, public check-ins etc, you must visit the individual beer's page
    
    # now save the details to an object and append to the list
    checkins.append({'checkin_id' : checkin_item_id.split('_')[1],
                     'beer_name' : beer_name,
                     'beer_url' : beer_url,
                     'brewery_name' : brewery_name,
                     'brewery_url' : brewery_url,
                     'venue_name' : venue_name,
                     'venue_url' : venue_url,
                     'rating' : rating,
                     'date' : date})

# all done, close the webdriver
driver.close()

# calculate the end time and the elapsed time
end_time = time.time()
print('end time {}'.format(dt.fromtimestamp(end_time).strftime('%H:%M:%S')))
print('elapsed time: {:,.1f} secs'.format(end_time-start_time))

# see my 10th check-in, as an example
checkins[-10]

# turn the list of check-in dicts into a dataframe
df = pd.DataFrame(checkins)
print('created {:,} rows'.format(len(df)))

# convert each timestamp to pacific time
def parse_convert_date(date_string):
    date_time = date_parser.parse(date_string)
    date_time_tz = date_time.replace(tzinfo=date_time.tzinfo).astimezone(pytz.timezone('US/Pacific'))
    return date_time_tz
    
df['date_pacific_tz'] = df['date'].map(parse_convert_date)
df = df.drop('date', axis=1)

df.head()

# save the dataset to csv
df.to_csv('data/untappd.csv', index=False, encoding='utf-8')



