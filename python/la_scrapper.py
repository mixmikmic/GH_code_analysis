from bs4 import BeautifulSoup
import requests
import re
import json
import scrapping_functions as sf
# may need pip install urllib3
import urllib3

urllib3.disable_warnings()
target_url = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc=Los_Angeles+CA&start='
base = 'http://www.yelp.com'

#Pull in a list of links from the target url
link_dict = {}
for x in range(10, 990, 10):
    target = target_url + str(x)
    raw_html = requests.get(target, verify=False)
    soup = BeautifulSoup(raw_html.text, 'html.parser')
    link_dict = sf.biz_links(soup, link_dict)

print("\nFinished!")

#Write all links to a text file
biz_links = open('cleanbiz_la_links.txt', 'w')
for item in link_dict.keys():
    if "adredir" in item: 
        continue
    print("key is: " + item)
    biz_links.write("%s\n" % item)
biz_links.close()
print("\nFinished writing to file:  cleanbiz_la_links.txt")

link_file = open("cleanbiz_la_links.txt", "r")
link_list = link_file.read().split('\n')
link_list = list(set(link_list))
for link in link_list:
    if link == '':
        link_list.pop(link_list.index(link))

biz_dict = {}

for biz_name in link_list:
    biz_dict[biz_name] = {}
    raw_html = requests.get(base + biz_name, verify=False)
    print(base + biz_name)
    soup = BeautifulSoup(raw_html.text, 'html.parser')
    biz_dict[biz_name] = json.loads(soup.find('script', type='application/ld+json').text)
    
print("\nFinished")

#Output JSON file of all the review details
with open('la_reviews.json', 'w') as outfile:
    json.dump(biz_dict, outfile)
print("Finished")



