import urllib.request  
from bs4 import BeautifulSoup 
import pandas as pd

page_url = 'http://rtqa.royalsurrey.nhs.uk/admin/qa/test/?type__exact=composite&p=5&o=4'
page =  urllib.request.urlopen(page_url) 

soup = BeautifulSoup(page, 'html.parser') 

soup

soup.title

soup.title.string

soup.title.parent.name

soup.p

soup.a

soup.a['class']

soup.find_all('a')

for link in soup.find_all('a'):
    print(link.get('href'))

print(soup.get_text())

tables = soup.findAll('table')
test = pd.io.html.read_html(str(tables))

return(test)            #return dataframe type object

from pandas.io.html import read_html
from selenium import webdriver


#driver = webdriver.Firefox()
driver = webdriver.Chrome()
driver.get(page_url)

table = driver.find_element_by_xpath('//div[@class="sp5"]/table//table/..')
table_html = table.get_attribute('innerHTML')

df = read_html(table_html)[0]
print(df)

driver.close()



