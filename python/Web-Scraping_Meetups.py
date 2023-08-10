#Importing the standard packages 
from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
import numpy as np

### This is a test for one event
### Allowing me to develop my webscraping skills 

meetup_path = requests.get("https://www.meetup.com/StormYoga/").text
yogatest_html = BeautifulSoup(meetup_path, "lxml")
#print yoga_meetup.title # find the title tags
#print yoga_meetup.title.string  # find the value of tags
location = yogatest_html.find_all('dd', {'class':'text--secondary'})
#print location
address = [place.get_text() for place in location]
#print address
group_info = yogatest_html.find_all('span',{'class':'lastUnit align-right'})
print group_info[0].get_text()
group_members = int(group_info[0].get_text().replace(',','')) 
group_review = int(group_info[1].get_text())
upcoming_meetings = int(group_info[2].get_text())
past_meetups = int(group_info[3].get_text())

#Key Words 

key_words = yogatest_html.find_all('div', {'id':re.compile('^topicList')})
key_word_tag =[word.find_all('a') for word in key_words]
#key_word_tags[0][0].get_text()
key_word_list = [tag.get_text() for tag in key_word_tag[0]]
print key_word_list




