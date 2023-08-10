import urllib.request as urllib2
from bs4 import BeautifulSoup

baseLink = "https://www.indeed.com"
link1 = "https://www.indeed.com/resumes?q=machine+learning&l=Mumbai%2C+Maharashtra&cb=jt"
link2 = "https://www.indeed.com/r/Rahul-Tripathi/3cf9ec868b63257b?sp=0"
page1 = urllib2.urlopen(link1)
page2 = urllib2.urlopen(link2)
soup1 = BeautifulSoup(page1, 'html.parser')
soup2 = BeautifulSoup(page2, 'html.parser')

jumpLinks = soup1.find_all('a', class_ = "app_link") 
print(len(jumpLinks))
i=0

for link in jumpLinks:
#     print(link.get('href'))
    print(link)
    CVlink = baseLink + link.get('href')
    print(CVlink)
    CVpage = urllib2.urlopen(CVlink)
    CVsoup = BeautifulSoup(CVpage, 'html.parser')

jumpLinks[2]

### 2nd PAGE
soup = soup2
print(soup.find(id="headline").string)
print(soup.find(id="headline_location").string)
print(soup.find(id="res_summary"))
print(soup.find_all('div', class_ = "section_title")[0].string) ## WORK EXPERIENCE TITLE  ####  USE FOR LOOP here !!!
print(soup.find('div', class_ = "work_company").string)  ## working company
print(soup.find('p', class_ = "work_dates").string)  ## Working duration
print(soup.find('p', class_ = "edu_title").string)  ## Bachleor info.
print(soup.find('div', class_ = "edu_school").string)  ## Bachleor colllege
print(soup.find_all('span', class_ = "skill-text")[0].string)   ## We have to use for loop here



