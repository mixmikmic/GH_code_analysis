import requests
from bs4 import BeautifulSoup
import os

html_target = "a"
tag = "href"
f_ext = ".pdf"
dir_name = "Ghodsi_Ali"
url = 'https://www.cs.berkeley.edu/~alig/papers'

### Request and Collect


r = requests.get(url)

status = r.status_code
encoding = r.encoding
html_doc = r.text

soup = BeautifulSoup(html_doc, 'html.parser')
anchor = soup(html_target)

def make_dir(directory):
    """
    return: None
    Makes directory if does not already exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_url(url, endpoint):
    """
    return: None
    downloads file, requires url in global or class scope.
    """
    url_addr = "{url}/{endpoint}".format(url=url, endpoint=endpoint)
    file_path = "{directory}/{endpoint}".format(directory=dir_name, endpoint=endpoint)
    
    r = requests.get(url_addr)
    content_file = r.content
    
    with open(file_path, 'wb') as f:
        print """Downloading From: {url}\nWriting to: {file_path}""".format(
                                                url=url_addr, 
                                                file_path=file_path
                                                                    )
        f.write(content_file)
    

print """Status: {status}\nEncoding: {encoding}""".format(status=status, 
                                                    encoding=encoding)
print "Begin downloading"

make_dir(dir_name)
for a in anchor:
    endpoint = a[tag]
    if endpoint[-4:] == f_ext:
            download_url(url, endpoint)
            print "Finished Download -- {tag}".format(tag=endpoint)
    #print "miss: {tag}".format(tag=endpoint)
    
print "Finished Downloading"

