get_ipython().system('pip install unicodecsv')

import os
from lxml import html
import requests
import unicodecsv as csv
import argparse


def parse(zipcode, filter=None):
    if filter == "newest":
        url = "https://www.zillow.com/homes/for_sale/{0}/0_singlestory/days_sort".format(zipcode)
    elif filter == "cheapest":
        url = "https://www.zillow.com/homes/for_sale/{0}/0_singlestory/pricea_sort/".format(zipcode)
    else:
        url = "https://www.zillow.com/homes/for_sale/{0}_rb/?fromHomePage=true&shouldFireSellPageImplicitClaimGA=false&fromHomePageTab=buy".format(
            zipcode)

    for i in range(5):
        # try:
        headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate, sdch, br',
            'accept-language': 'en-GB,en;q=0.8,en-US;q=0.6,ml;q=0.4',
            'cache-control': 'max-age=0',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        print(response.status_code)
        parser = html.fromstring(response.text)
        search_results = parser.xpath("//div[@id='search-results']//article")
        properties_list = []

        for properties in search_results:
            raw_address = properties.xpath(".//span[@itemprop='address']//span[@itemprop='streetAddress']//text()")
            raw_city = properties.xpath(".//span[@itemprop='address']//span[@itemprop='addressLocality']//text()")
            raw_state = properties.xpath(".//span[@itemprop='address']//span[@itemprop='addressRegion']//text()")
            raw_postal_code = properties.xpath(".//span[@itemprop='address']//span[@itemprop='postalCode']//text()")
            raw_price = properties.xpath(".//span[@class='zsg-photo-card-price']//text()")
            raw_info = properties.xpath(".//span[@class='zsg-photo-card-info']//text()")
            raw_broker_name = properties.xpath(".//span[@class='zsg-photo-card-broker-name']//text()")
            url = properties.xpath(".//a[contains(@class,'overlay-link')]/@href")
            raw_title = properties.xpath(".//h4//text()")

            address = ' '.join(' '.join(raw_address).split()) if raw_address else None
            city = ''.join(raw_city).strip() if raw_city else None
            state = ''.join(raw_state).strip() if raw_state else None
            postal_code = ''.join(raw_postal_code).strip() if raw_postal_code else None
            price = ''.join(raw_price).strip() if raw_price else None
            info = ' '.join(' '.join(raw_info).split()).replace(u"\xb7", ',')
            broker = ''.join(raw_broker_name).strip() if raw_broker_name else None
            title = ''.join(raw_title) if raw_title else None
            property_url = "https://www.zillow.com" + url[0] if url else None
            is_forsale = properties.xpath('.//span[@class="zsg-icon-for-sale"]')
            properties = {
                'address': address,
                'city': city,
                'state': state,
                'postal_code': postal_code,
                'price': price,
                'facts and features': info,
                'real estate provider': broker,
                'url': property_url,
                'title': title
            }
            if is_forsale:
                properties_list.append(properties)
        return properties_list
    # except:
    #   print ("Failed to process the page",url)

def scrape_all_sf_area_codes(output_folder):
    """Scrape from all SF area codes"""
    zipcodes = [94102,
             94104,
             94103,
             94105,
             94108,
             94107,
             94110,
             94109,
             94112,
             94111,
             94115,
             94114,
             94117,
             94116,
             94118,
             94121,
             94123,
             94122,
             94124,
             94127,
             94126,
             94129,
             94131,
             94133,
             94132,
             94134,
             94139,
             94143,
             94146,
             94151,
             94159,
             94158,
             94188,
             94177,
             ]
    sort = 'newest'
    for zipcode in zipcodes:
        print ("Fetching data for %s" % (zipcode))
        scraped_data = parse(str(zipcode), sort)
        print ("Writing data to output file")
        with open(f"{output_folder}/properties-{zipcode}.csv", 'wb')as csvfile:
            fieldnames = ['title', 'address', 'city', 'state', 'postal_code', 'price', 'facts and features',
                          'real estate provider', 'url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in scraped_data:
                writer.writerow(row)
def run(output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    scrape_all_sf_area_codes(output_folder=output_folder)

run(output_folder='./data/sf/mar9_2018/')



