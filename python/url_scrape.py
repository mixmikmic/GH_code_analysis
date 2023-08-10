from requests import post
from requests import get
from bs4 import BeautifulSoup

import math
from csv import writer

def scrape_per_year(year, total_number):
    
    masterURL_list = []
    
    pages = (math.ceil(total_number / 100) + 1)
    
    for i in range(1,pages):
        
        print("Scraping page: ", i)

        url = "https://foiaonline.regulations.gov/foia/action/public/search//runSearch"

        # Data we're sending to the server
        input_data = {
        "searchParams.searchTerm": str(year),
        "searchParams.forRequest" : "true",
        "searchParams.allAgencies" : "true",
        "d-5509183-p": i,
        "pageSize": 100,
        "_sourcePage" : "OqOjCyYuqKf67cq6ztrYwRQ5liT1byr9zIr-khSH2rDr8C-REYyxgQ==",
        "__fp" : "gxy8iASbrd4axWlNcO6xThSknnzmCxPr3Aq5U6uHZtUaxWlNcO6xTou453r9KOlmYzl23PVmFIzkb_mqipWoBvG3hsCkh3fRlH7bCujMItrKE90c771wQcUBlJjAHbVa2kCOX_ImUm6JoBvHmEb2fuRv-aqKlagG8beGwKSHd9GS726XiOL27_HOqbOv1VyM-qtcGtit4tseLbMAO4pNGA=="
        }

        # The browser we are pretending to be
        head_data = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
                 "From":"rsk2161@columbia.edu"}

        response = post(url,data=input_data,headers=head_data)    
        data = response.text

        soup = BeautifulSoup(data, 'html.parser')

        results_table = soup.find("table", {"id":"dttPubSearch"})
        table_body = results_table.find("tbody")

        all_links = table_body.find_all("a")

        cleaned_links = []
        for item in all_links:
            cleaned_links.append(item["href"])

        shortened_linklist = cleaned_links[0: :2]
        masterURL_list.extend(shortened_linklist)

    print("++++++++ TOTAL LINKS ++++++++")
    print(len(masterURL_list))
    new_list = [masterURL_list]
    
    csv_name = "foia_urls-" + str(year) + ".csv"
    outfile = writer(open(csv_name, "w"))
    
    outfile.writerow(new_list)
    
    return 

scrape_per_year(2011, 24604)

scrape_per_year(2012, 42466)

scrape_per_year(2013, 62698)

scrape_per_year(2014, 82097)

scrape_per_year(2015, 116817)

scrape_per_year(2016, 360474)

scrape_per_year(2017, 27462)

# spamReader = csv.reader(open('eggs.csv', newline=''), delimiter=' ', quotechar='|')

f = open("foia_urls-2011.csv", "r")
data=f.read()

urls = data.split(",")

f = open("foia_urls-2011.csv", "r")
data=f.read()

urls = data.split(",")


    

def url_cleaner(record_year):
    filename = "foia_urls-" + str(record_year) + ".csv"
    f = open(filename, "r")
    data=f.read()

    urls = data.split(",")
    
    new_filename = str(record_year) + "-urls.txt"

    with open(new_filename, "w") as text_file:
        for item in urls:
            cleaned = (item.strip())[1:-1]
            base = "https://foiaonline.regulations.gov"
            new = base + cleaned
            print(new, file=text_file)

    f.close()

# 24604
url_cleaner(2011)

# 42466
url_cleaner(2012)

# 62698
url_cleaner(2013)

# 82097
url_cleaner(2014)

# 116817
url_cleaner(2015)

# 360474
url_cleaner(2016)

# 27500
url_cleaner(2017)



