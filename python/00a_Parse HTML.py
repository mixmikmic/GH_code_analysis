import os
import pandas as pd
from lxml import html
import gzip

get_ipython().magic('pwd')

html_folder = './html_women'

# Reading Julian's files
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# read input file and use as master list to find html files
df = pd.read_csv('asin_Women.csv', header=None)
print df.columns
print df.shape

meta = getDF('meta_Clothing_Shoes_and_Jewelry.json.gz')
print meta.columns
print meta.shape

# create new dataframe from meta slice
if meta is not None:
    c5 = meta[meta.asin.isin(df[0].tolist())].reset_index().copy()
    meta = None

c5.head()

# count of missing price/brand
print c5.dtypes
print '\n'
print 'length:', c5.shape[0]
print 'missing price:', c5[(c5.price.isnull()) | (c5.price == -1)].shape[0]
print 'missing brand:', c5[(c5.brand == '') | (c5.brand.isnull()) | (c5.brand == 'Unknown')].shape[0]
print 'missing description', c5[(c5.description == '') | (c5.description.isnull())].shape[0]

def extract(asin):
    def getDicFromTable(html):
        # extract only html nodes from raw html
        rows = [x for x in html if str(type(x)) == "<class 'lxml.html.HtmlElement'>"]

        # return dictionary from list of tuples of headers and data elements
        return dict(
            zip([x.text.strip() for x in rows if x.tag == 'th'], [x.text.strip() for x in rows if x.tag == 'td']))

    def getProductInfo(raw):
        # parse tabular list and return dictionary
        rows = [x for x in raw if str(type(x)) == "<class 'lxml.html.HtmlElement'>"]

        mydic = dict()
        mylist = [x for x in rows if x.tag == 'li']

        for li in mylist:
            if len(li.getchildren()) == 1:
                mydic[li[0].text.strip()] = li[0].tail.strip() if li[0].tail is not None else ''
            else:
                mydic[li[0].text.strip()] = li[0].tail.strip() + ''.join([x.text_content().strip() + (x.tail.strip() if x.tail is not None else '') for x in li[1:] if x.tag not in ['ul','style','script']])
        return mydic
    
    source =  u''.join([x.decode('latin-1') for x in open( (html_folder + '/{asin}.html').format(asin=asin), 'r')])
    doc = html.fromstring(source)

    XPATH_NAME = '//h1[@id="title"]//text()'
    XPATH_SALE_PRICE = '//span[contains(@id,"ourprice") or contains(@id,"saleprice")]/text()'
    XPATH_ORIGINAL_PRICE = '//td[contains(text(),"List Price") or contains(text(),"M.R.P") or contains(text(),"Price")]/following-sibling::td/text()'
    XPATH_BRAND = '//a[@id="brand"]//text()'
    XPATH_BRAND_img = '//a[@id="brand"]/@href'
    XPATH_FEATURE_BULLETS = '//div[@id="feature-bullets"]//li/span[@class="a-list-item"]/text()'
    XPATH_PRODUCT_INFORMATION = '//table[@id="productDetails_detailBullets_sections1"]//tr/node()'
    XPATH_PRODUCT_INFO_li = '//div[@id="detailBullets_feature_div"]//li//text()'
    XPATH_PRODUCT_INFO_div2 = '//div[@id="detail-bullets"]//div[@class="content"]/ul/node()'
    XPATH_PRODUCT_DESCRIPTION = '//div[@id="productDescription"]//text()'
    XPATH_PRODUCT_DESC2 = '//div[@id="productDescription"]//text()'
    
    RAW_NAME = doc.xpath(XPATH_NAME)
    RAW_SALE_PRICE = doc.xpath(XPATH_SALE_PRICE)
    RAW_ORIGINAL_PRICE = doc.xpath(XPATH_ORIGINAL_PRICE)
    RAW_BRAND = doc.xpath(XPATH_BRAND)
    RAW_BRAND_img = doc.xpath(XPATH_BRAND_img)
    RAW_FEATURE_BULLETS = doc.xpath(XPATH_FEATURE_BULLETS)
    RAW_PRODUCT_INFORMATION = doc.xpath(XPATH_PRODUCT_INFORMATION)
    RAW_PRODUCT_INFO_li = doc.xpath(XPATH_PRODUCT_INFO_li)
    RAW_PRODUCT_INFO_div2 = doc.xpath(XPATH_PRODUCT_INFO_div2)
    RAW_PRODUCT_DESCRIPTION = doc.xpath(XPATH_PRODUCT_DESCRIPTION)
    RAW_PRODUCT_DESC2 = doc.xpath(XPATH_PRODUCT_DESC2)
    
    NAME = ' '.join(''.join(RAW_NAME).split()) if RAW_NAME else None
    SALE_PRICE = ' '.join(''.join(RAW_SALE_PRICE).split()).strip() if RAW_SALE_PRICE else None
    ORIGINAL_PRICE = ''.join(RAW_ORIGINAL_PRICE).strip() if RAW_ORIGINAL_PRICE else None
    BRAND = ''.join(RAW_BRAND).strip()
    BRAND_img = None
    if len(RAW_BRAND_img) > 0:
        BRAND_img = ' '.join(RAW_BRAND_img[0].split('=')[-1].split('+'))
    
    FEATURE_BULLETS = [x.strip() for x in RAW_FEATURE_BULLETS if x.strip() != '']
    PRODUCT_INFORMATION = getDicFromTable(RAW_PRODUCT_INFORMATION)
    PRODUCT_INFO_li = [x.strip() for x in RAW_PRODUCT_INFO_li if x is not None and x.strip() != '']
    PRODUCT_INFO_div2 = getProductInfo(RAW_PRODUCT_INFO_div2)
    PRODUCT_DESCRIPTION = '\n'.join([x.strip() for x in RAW_PRODUCT_DESCRIPTION if x.strip() != ''])
    PRODUCT_DESC2 = ' '.join([x.strip() for x in RAW_PRODUCT_DESC2 if x.strip() != ''])

    if not ORIGINAL_PRICE:
        ORIGINAL_PRICE = SALE_PRICE
    
    if PRODUCT_INFORMATION == False:
        PRODUCT_INFORMATION = dict(zip(PRODUCT_INFO_li[::2],PRODUCT_INFO_li[1::2])) if len(PRODUCT_INFO_li) > 0 else PRODUCT_INFO_div2
       
    
    return {
            'asin' : asin,
            'NAME' : NAME,
            'SALE_PRICE': SALE_PRICE,
            'ORIGINAL_PRICE' : ORIGINAL_PRICE,
            'BRAND' : BRAND_img if BRAND is None or BRAND == '' else BRAND,
            'PRODUCT_INFORMATION' : PRODUCT_INFORMATION,
            'FEATURE_BULLETS' : FEATURE_BULLETS,
            'PRODUCT_DESCRIPTION' : PRODUCT_DESC2 if PRODUCT_DESCRIPTION == '' or PRODUCT_DESCRIPTION is None else PRODUCT_DESCRIPTION
           }

# extract features from html files
files = [x for x in os.listdir(html_folder) if x.endswith('.html')]
scraped = [extract(x[:-5]) for x in files]

# merge existing and scraped features
combo = pd.merge(c5, pd.DataFrame.from_records(scraped), on='asin', how='left').drop('index', axis='columns')
combo.shape

# save to file
combo.to_hdf('womens_Meta_scraped.hd5', key='data', compression='blosc')

