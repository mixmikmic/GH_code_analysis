from bs4 import BeautifulSoup as bs
import urllib

# set the url we want to visit
url = "https://www.bloomberg.com/asia"

# A:
contents = urllib.urlopen(url).read()

len(contents)

# A:
contents[:100000]

# A:
html_obj = bs(contents, 'html.parser', from_encoding='utf-8')

# A:
print html_obj.find_all('a', {'class': 'hero-v6-story__headline-link'})

# A:
a_names = []
# for each element you find, print out the article name
for article in html_obj.find_all('a', {'class': 'hero-v6-story__headline-link'}):
    title = article.renderContents()
    a_names.append(title.strip())

# A:
for article in html_obj.find_all('a', {'class': 'highlights-v6-story__headline-link'}):
    title = article.renderContents()
    a_names.append(title.strip())

a_names

# A:
# Get homepage link then get price change worse
asset_classes = html_obj.find_all('iframe', {'class': 'market-summary-v3'})

asset_classes



# A:

# A:

# import
from selenium import webdriver

# create a driver called driver
driver = webdriver.Chrome(executable_path="./chromedriver/chromedriver")

# close it
driver.close()

# A:
driver = webdriver.Chrome(executable_path="./chromedriver/chromedriver")
driver.get("https://www.bloomberg.com/asia")

# A:
assert 'Bloomberg - Asia Edition' in driver.title

# import sleep
from time import sleep

# A:
driver = webdriver.Chrome(executable_path="./chromedriver/chromedriver")
driver.get("https://www.bloomberg.com/asia")
sleep(20)
blmberg_content = driver.page_source

# there is a difference between scraping from selenium and urllib
len(blmberg_content)

# convert to beautiful soup first
blmberg_soup = BeautifulSoup(blmberg_content, 'lxml')

# get asset class summaries on page
asset_classes = blmberg_soup.find_all('iframe', {'class':'market-summary-v3'})

asset_classes

# A:

# A:

# A:

# A:

# we can send keys as well
# from selenium.webdriver.common.keys import Keys

# # open the driver
# driver = webdriver.Chrome(executable_path="./chromedriver/chromedriver")
# # visit Python
# driver.get("http://www.python.org")
# # verify we're in the right place
# assert "Python" in driver.title

# # find the search position
# elem = driver.find_element_by_name("q")
# # clear it
# elem.clear()
# # type in pycon
# elem.send_keys("pycon")

# # send those keys
# elem.send_keys(Keys.RETURN)
# # no results
# assert "No results found." not in driver.page_source

# driver.close()

# # all at once:
# driver = webdriver.Chrome(executable_path="./chromedriver/chromedriver")
# driver.get("http://www.python.org")
# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
# driver.close()

