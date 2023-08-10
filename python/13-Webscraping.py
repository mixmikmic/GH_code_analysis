import json
import requests
from bs4 import BeautifulSoup

from IPython.display import Image

Image(filename='figs/apple1.png')

# Search Parameters
search_request = {
        'searchString': '',
        'jobType': 0,
        'sortBy': 'req_open_dt',
        'sortOrder': '1',
        'language': None,
        'autocomplete': None,
        'delta': 0,
        'numberOfResults': 0,
        'pageNumber': None,
        'internalExternalIndicator': 0,
        'lastRunDate': 0,
        'countryLang': None,
        'filters': {
              'locations':{
                   'location':[{
                            'type': 0,
                            'code': 'USA',
                            'countryCode': None,
                            'stateCode': None,
                            'cityCode': None,
                            'cityName': None
                            }]
                        },
                    'languageSkills': None,
                    'jobFunctions': None,
                    'retailJobSpecs': None,
                    'businessLine': None,
                    'hiringManagerId': None},
                'requisitionIds': None
        }

# Scraping method
def scrape_responses(max_pages=2):
    pageno = 0
    search_request['pageNumber'] = pageno

    myresponses = []
    while pageno < max_pages:
        payload = {
                    'searchRequestJson': json.dumps(search_request),
                    'clientOffset': '-300'
                    }
        resp = requests.post(
                url = 'https://jobs.apple.com/us/search/search-result',
                data = payload,
                headers = {
                    'X-Requested-With': 'XMLHttpRequest'
                    }
                )
        myresponses.append(resp)
        
       
        pageno += 1
        search_request['pageNumber'] = pageno
        
    return myresponses

def responses_to_jobs(responses):
    jobs = []
    for resp in responses:
         # BeautifulSoup parser
        s = BeautifulSoup(resp.text, 'lxml')

    #        if not s.requisition:
    #            break
        
        for r in s.findAll('requisition'):
            job = {}
            job['jobid'] = r.jobid.text
            job['title'] = r.postingtitle and                     r.postingtitle.text or r.retailpostingtitle.text
            job['location'] = r.location.text
            jobs.append(job)

        return jobs

def scrape_apple_jobs():
    the_responses = scrape_responses()
    jobs = responses_to_jobs(the_responses)
    for job in jobs:
        print (job)

scrape_apple_jobs()

from selenium import webdriver
from time import sleep

# The path should lead to your copy of chromedriver
driver = webdriver.Chrome('/Users/asmith/repositories/github/BU-CS506-Spring2018/chromedriver')
 

# YouTube search URL
url = "https://www.youtube.com/results?q=formula+1"

# Open the page in Selenium driver instance
driver.get(url)

Image(filename="figs/youtube2.png")

def get_youtube_videos(driver):
    # Use XPath to get all HTML elements corresponding to videos
    videos = driver.find_elements_by_xpath("""//*[@id="video-title"]""")
    
    # Get video attributes
    for video in videos:
        title = video.get_attribute('title')
        href = video.get_attribute('href')
        
        print (title)
        print (href)
        print ("-----")

get_youtube_videos(driver)

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

actions = ActionChains(driver)

for _ in range(5):
    actions.send_keys(Keys.PAGE_DOWN).perform()
    sleep(5)

get_youtube_videos(driver)



