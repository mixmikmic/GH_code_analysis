get_ipython().system('pip install scrapy')

import scrapy
class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'http://quotes.toscrape.com/page/1/',
            'http://quotes.toscrape.com/page/2/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

get_ipython().system('pip install selenium')

chromedriverpath = '/Users/jonathan/Code/trial.ai/chromedriver'

css_cat_path = 'https://www.chowsangsang.com/tc/product/Jewellery-ByCategory-Charms?page=0'
css_product_path = 'https://www.chowsangsang.com/eshop-hk/zh_HK/%E8%B2%A8%E5%93%81/%E5%93%81%E7%89%8C%E7%B3%BB%E5%88%97/%E6%97%97%E8%89%A6%E7%B3%BB%E5%88%97/CHARME%E7%B3%BB%E5%88%97/Charme/p/PRD-90019GFC-489828?gaSrc=%E4%B8%B2%E9%A3%BE'

cartier_cat_path = 'http://www.cartier.com/en-us/collections/jewelry/collections/juste-un-clou.viewall.html'
cartier_product_path = 'http://www.cartier.com/en-us/collections/jewelry/exceptional-creations/high-jewelry-fauna-&-flora.html'

from selenium import webdriver

browser = webdriver.Chrome(chromedriverpath) #replace with .Firefox(), or with the browser of your choice
url = css_cat_path
browser.get(url) #navigate to the page

innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string

innerHTML.find('Charme')

innerHTML[44640:44646]

len(innerHTML)

print("There is another package for scraping and parsing data.")
print("The name is BeautifulSoup.")



