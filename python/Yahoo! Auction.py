import requests
import bs4
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_id(page_src, id_list):
    # check all items available in this page, and add id to id_list if
    # the time left for auction is greater than 5 mins (so that we can
    # get the original time for auction to be closed) (to keep enough
    # time to crawl all the original auction expiration time, we exclude items
    # with 10 mins left, too.)

    soup = bs4.BeautifulSoup(page_src, "lxml")
    item_table = soup.find("table")
    item_list = item_table.findAll("tr")

    for i in range(len(item_list)):
        try:
            time_left = item_list[i].find("td", {"class": "ti"}).text
            if time_left in ['1分','2分','3分','4分','5分','10分']:
                continue
            id_ = item_list[i+2].find("a", {"class": "b unwt"})['id'].split(':')[0]
            id_list.append(id_)
        except:
            None

def get_due_time(item_page):
    
    soup = bs4.BeautifulSoup(item_page, "lxml")

    for row in soup.findAll("li", {"class": "ProductDetail__item"}):
        if row.find("dt", {"class": "ProductDetail__title"}).text == "終了日時":
            return row.find("dd").text.split("：")[1]

# initialize web driver
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

driver.get('https://auctions.yahoo.co.jp/user/jp/show/mystatus')

# log in Yahoo!
# fill the usrname and password
username = driver.find_element_by_id("username")
username.send_keys(usrname)
driver.find_element_by_id("btnNext").click()

password = driver.find_element_by_id("passwd")
password.send_keys(password)
driver.find_element_by_id("btnSubmit").click()

driver.get('https://auctions.yahoo.co.jp/category/list/%E3%83%91%E3%82%BD%E3%82%B3%E3%83%B3-%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF-%E3%82%AA%E3%83%BC%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3/2084039759/?p=%E3%83%91%E3%82%BD%E3%82%B3%E3%83%B3&tab_ex=commerce&s1=end&o1=a')
id_list = []

get_id(driver.page_source, id_list)
for i in range(30):
    next_page = driver.find_element_by_class_name("next").find_element_by_tag_name("a").get_attribute("href")
    driver.get(next_page)
    get_id(driver.page_source, id_list)

len(id_list)

item_url = 'https://page.auctions.yahoo.co.jp/jp/auction/'
items = {}

for id_ in id_list:
    items[id_] = {}
    driver.get(item_url + id_)
    items[id_]['due_time'] = get_due_time(driver.page_source)

driver.close()

# change the due time to time stamp
for item in items:
    a = items[item]['due_time']
    try:
        timeArray = time.strptime(a, "%Y.%m.%d（" + a.split('（')[1][0] + "）%H:%M")
        timeStamp = int(time.mktime(timeArray))
        items[item]['due_timeStamp'] = timeStamp
    except:
        None

def get_log(id_):
    # get the log of bids
    
    log_list = []
    log_url = 'https://auctions.yahoo.co.jp/jp/show/bid_hist?aID={0}&apg={1}&typ=log#listtop'
    driver.get(log_url.format(id_, 1))
    soup = bs4.BeautifulSoup(driver.page_source, "lxml")
    log_table = soup.findAll("tbody")[5]
    log_list += log_table.findAll("tr")
    
    bid_number = int(driver.find_element_by_class_name("pts03").find_element_by_tag_name("p").text.split('（')[1].split('：')[1].split('件')[0])
    page_number = bid_number // 50 + 1
    
    for page in range(page_number-1):
        driver.get(log_url.format(id_, page+2))
        soup = bs4.BeautifulSoup(driver.page_source, "lxml")
        log_table = soup.findAll("tbody")[5]
        log_list += log_table.findAll("tr")
    
    log_list = [log.text for log in log_list]
    
    return log_list

# initialize web driver
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

driver.get('https://auctions.yahoo.co.jp/user/jp/show/mystatus')

# log in Yahoo!
username = driver.find_element_by_id("username")
username.send_keys(usrname)
driver.find_element_by_id("btnNext").click()

password = driver.find_element_by_id("passwd")
password.send_keys(password)
driver.find_element_by_id("btnSubmit").click()

for item in items:
    try:
        items[item]['log_list'] = get_log(item)
    except:
        None

driver.close()

valid_auction = {}
for item in items:
    
    try:
        bids_number = len(items[item]['log_list'])
        if bids_number <= 1:
            continue
        valid_auction[item] = items[item]
    except:
        None

len(valid_auction)

# find out the auctions which is extended
extended_auction = {}

for item in valid_auction:
    a = valid_auction[item]['log_list'][0].split('\n')[1]
    timeArray = time.strptime("2017" + a, "%Y[%m月 %d日 %H時 %M分]")
    timeStamp = int(time.mktime(timeArray))
    
    if timeStamp > valid_auction[item]['due_timeStamp']:
        extended_auction[item] = valid_auction[item]

len(extended_auction)

with open('/Users/hyde/Dropbox/auctions/Yahoo!_Auction/auction_log.txt', 'w') as f:
    for item in extended_auction:
        f.write('ID:{}\n'.format(item))
        f.write('expiration time:{}\n'.format(items[item]['due_time']))
        f.write('-'*40 + '\n\n')
        f.write('time' + ' '*4 + 'user' + ' '*5 + 'bid\n')
        f.write('-'*40 + '\n\n')
        for line in items[item]['log_list']:
            f.write(' '.join(line.split('\n')) + '\n')
        
        f.write('='*80 + '\n\n')



