from urllib.request import urlopen
#NOTE Python 2 users use this syntax 
#from urllib2 import urlopen

#connecting to a webpage
html = urlopen("https://www.indeed.com/viewjob?jk=d88f3bb95f1a4466")

print(html)

from bs4 import BeautifulSoup 

#Creating the python object to connect to the site
site = html.read()

print(site)

#Creating BeautifulSoup object to make parsing raw html easier
soup = BeautifulSoup(site, 'html.parser')

print(soup.prettify())

#This returns ONLY the first tag
print(soup.meta)

#Removing an individual tag from the BeautifulSoup object
soup.meta.decompose()

print(soup.prettify())    

#removing all the formatting and meta tags from the object
for i in soup.find_all("script"):
    i.decompose()
for i in soup.find_all("style"):
    i.decompose()
for i in soup.find_all("noscript"):
    i.decompose()
for i in soup.find_all("div"):
    i.decompose()
for i in soup.find_all("meta"):
    i.decompose()
    
print(soup.prettify())

#getting only the text that would be visible on the webpage
text = soup.get_text()

print(text)

# breaking text into lines and removing excess whitespace
lines = [line.strip() for line in text.splitlines()]
lines = [l for l in lines if l != '']

for l in lines:
    print(l)

#flattening onto one line
cleaned = ' '.join(lines)

print(cleaned)

#function to encapsulate logic above to connect to a page and pull the job posting
def get_text(url):
    html = urlopen(url)
    
    site = html.read()
    soup = BeautifulSoup(site, 'html.parser')
    for i in soup.find_all("script"):
        i.decompose()
    for i in soup.find_all("style"):
        i.decompose()
    for i in soup.find_all("noscript"):
        i.decompose()
    for i in soup.find_all("div"):
        i.decompose()
    for i in soup.find_all("meta"):
        i.decompose()
    text = soup.get_text()
    # break into lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l != '']
    cleaned = ' '.join(lines)
    return(cleaned)

#testing to make sure it works
get_text("https://www.indeed.com/viewjob?jk=d88f3bb95f1a4466")

url = 'https://www.indeed.com/jobs?q=data+scientist'
html = urlopen(url)
    
site = html.read()
soup = BeautifulSoup(site, 'html.parser')
print(soup.prettify())

# accessing the results section of the page
results = soup.find(id = 'resultsCol')

print(results)

# getting an array of all the urls contained within the results
print(results.find_all('a'))

#cleaning it up to only urls
page_urls = [link.get('href') for link in results.find_all('a')]

print(page_urls)

#keeping only strings that contain '/rc/'
page_urls = [link for link in page_urls if '/rc/' in str(link)]

print(page_urls)

#array to store job ids
ids = []
for link in page_urls:
    #finding locations in the string to start and stop
    start = link.find('jk=') + 3
    end = link.find('&fccid=')
    
    #appending id to the array
    ids.append(link[start:end])
    
print(ids)

#combining logic above to return the job postings from a single page
def get_links(url):
    html = urlopen(url)
    
    site = html.read()
    soup = BeautifulSoup(site, 'html.parser')
    results = soup.find(id = 'resultsCol')
    page_urls = [link.get('href') for link in results.find_all('a')]
    page_urls = [link for link in page_urls if '/rc/' in str(link)]

    
    ids = []
    for link in page_urls:
        start = link.find('jk=') + 3
        end = link.find('&fccid=')
        ids.append(link[start:end])
    return(ids)

get_links('https://www.indeed.com/jobs?q=data+scientist')

url = 'https://www.indeed.com/jobs?q=data+scientist'
html = urlopen(url)
site = html.read()
soup = BeautifulSoup(site, 'html.parser')
results = soup.find(id = 'resultsCol')

print(results)

#grabbing the id tag for the search count
count_string = soup.find(id = 'searchCount')

print(count_string)

#getting just the text in the string
count_string = count_string.get_text()
#spitting the string by white spaces and keeping only the last value
count = count_string.split(' ')[-1]
#removing the punctuation
count = count.replace(',', '')
#converting to a number
count = int(count)

print(count)

from time import sleep 

i = 0
job_ids = []
while i < 20:
    #creating custom search url for the correct page of results
    url = 'https://www.indeed.com/jobs?q=data+scientist&start=' + str(i)
    #calling function to grab ids and adding to the array
    job_ids = job_ids + get_links(url)
    #pausing for a second so we don't overload their server
    sleep(1)
    #updating to get to the next page
    i = i + 10    
    
print(job_ids)

descriptions = []

for id in job_ids:
    #generating custom job url
    url = "https://www.indeed.com/viewjob?jk=" + str(id)
    #appending to the end of the array
    descriptions.append(get_text(url))
    #pausing for a second so we don't overload their server
    sleep(1)
    
print(descriptions)

def get_job_descriptions(search_url):
    html = urlopen(search_url)
    site = html.read()
    soup = BeautifulSoup(site, 'html.parser')
    results = soup.find(id = 'resultsCol')

    divs = results.find_all('div')
    for x in divs:
        if x.get('id') == 'searchCount':
            count_string = x

        
    #getting just the text in the string
    count_string = count_string.get_text()
    #spitting the string by white spaces and keeping only the last value
    count = count_string.split(' ')[-1]
    #removing the punctuation
    count = count.replace(',', '')
    #converting to a number
    count = int(count)
    
    print("Getting job ids...")

    i = 0
    job_ids = []
    while i < count:
        #creating custom search url for the correct page of results
        url = search_url + '&start=' + str(i)
        #calling function to grab ids and adding to the array
        job_ids = job_ids + get_links(url)
        #pausing for a second so we don't overload their server
        sleep(1)
        #updating to get to the next page
        i = i + 10

    print("Finished pulling job ids. Getting descriptions...")
    
    descriptions = []
    
    #creating a counter to keep track of progress
    iteration_counter = 0
    for id in job_ids:
        #Printing out progress if we are on a tenth job id
        if iteration_counter % 10 == 0:
            print("Grabbing job " + str(iteration_counter))
        iteration_counter = iteration_counter + 1
        
        #generating custom job url
        url = "https://www.indeed.com/viewjob?jk=" + str(id)
        #appending to the end of the array
        descriptions.append(get_text(url))
        #pausing for a second so we don't overload their server
        sleep(1)
    
    return(descriptions)

#This will take a while!
job_descriptions = get_job_descriptions('https://www.indeed.com/jobs?q=data+scientist')

#looking at a few of the results
print(job_descriptions[40:50])

#saving results to a text file to reopen later if desired
f = open("job_descriptions.txt", "w")

for i in job_descriptions:
    f.write(str(i)+'\n')
    
#f.write("\n".join(map(lambda x: str(x), job_descriptions)))
f.close()

