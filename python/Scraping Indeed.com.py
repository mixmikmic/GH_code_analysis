# import libraries
from bs4 import BeautifulSoup as Soup
import requests
import pandas as pd

# indeed.com url
base_url = 'http://www.indeed.com/jobs?q=data+scientist&jt=fulltime&sort='
sort_by = 'date'          # sort by data
start_from = '&start='    # start page number
home_url = "http://www.indeed.com"
print(home_url)

# Create a list to contain all the job postings
job_listings = []

for page in range(0,500,10): # page from 1 to 100 (last page we can scrape is 100)
    if page % 100 == 0:
        print("Scraping page {}".format(page // 10))
    url = "%s%s%s%d" % (base_url, sort_by, start_from, page) # get full url
    target = Soup(requests.get(url).text, "lxml") 

    targetElements = target.findAll('div', attrs={'class' : 'result'}) # we're interested in each row (= each job)
    
    # trying to get each specific job information (such as company name, job title, urls, ...)
    for elem in targetElements:
        
        try:
            comp_name = elem.find('span', "company").text.strip()
            job_title = elem.find('a', attrs={'class':'turnstileLink'}).attrs['title']
            job_addr = elem.find('span',"location").text
            job_link = "%s%s" % (home_url,elem.find('a').get('href'))
            job_summary = elem.find('span',"summary").text.strip()

            if elem.find('span', "company").find("a"):

                company_link = elem.find('span', "company").find("a")
                comp_link_overall = "%s%s" % (home_url, company_link['href'])
            else:
                comp_link_overall = None

            # add a job info to our data frame
            job_listings.append({'company_name': comp_name, 
                                 'job_title': job_title, 
                                 'job_link': job_link,
                                 'job_summary': job_summary,
                                 'company_link': comp_link_overall, 
                                 'job_location': job_addr})
        
        # Some ofthe listings are missing information, we are going to skip them
        except:
            print("Bad data on search page")
            print(url)
            
print("Scrapting Finish! Collected {} job postings!".format(len(job_listings)))

jobs_dataframe = pd.DataFrame(job_listings)
jobs_dataframe.to_csv("jobs-data.csv", index=False)
jobs_dataframe.head()

# remove duplicate company URLs
company_urls = set(listing['company_link'] for listing in job_listings)
print(len(job_listings))
print(len(company_urls))

company_info = []

for i,company_url in enumerate(company_urls):
    if i % 50 == 0:
        total_listings = len(company_urls)
        print("Scraping company {} of {}".format(i, total_listings))
    
    # skip the None 
    if not company_url:
        continue

    company_page = Soup(requests.get(company_url).text, "lxml")
    
    
    # get the company ratings
    ratings = company_page.find_all('span','cmp-star-rating')
    
    company_info.append({
    'url'                          : company_url,
    'overall_rating'               : float(company_page.find('span','cmp-average-rating').text),
    'wl_balanace_rating'           : float(ratings[0].text),
    'compensation_benefits_rating' : float(ratings[1].text),
    'js_advancement_rating'        : float(ratings[2].text),
    'management_rating'            : float(ratings[3].text),
    'culture_rating'               : float(ratings[4].text)})
    
    
    

company_dataframe = pd.DataFrame(company_info)
company_dataframe.to_csv("company-data.csv", index=False)
company_dataframe.head()





