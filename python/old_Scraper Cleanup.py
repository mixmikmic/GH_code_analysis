import pandas as pd
get_ipython().magic('pylab inline')
from IPython.core.display import HTML
from lxml import html
import json
import os

get_ipython().magic('pwd')


def extract(asin):
    # this function parses values missed by the parser logic in the scraper
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
    source =  u''.join([x.decode('utf-8') for x in open('./html/{asin}.html'.format(asin=asin), 'rb')])
    doc = html.fromstring(source)

    XPATH_BRAND_img = '//a[@id="brand"]/@href'
    XPATH_PRODUCT_INFO_li = '//div[@id="detailBullets_feature_div"]//li//text()'
    XPATH_PRODUCT_INFO_div2 = '//div[@id="detail-bullets"]//div[@class="content"]/ul/node()'
    XPATH_PRODUCT_DESC2 = '//div[@id="productDescription"]//text()'
    
    RAW_BRAND_img = doc.xpath(XPATH_BRAND_img)
    RAW_PRODUCT_INFO_li = doc.xpath(XPATH_PRODUCT_INFO_li)
    RAW_PRODUCT_INFO_div2 = doc.xpath(XPATH_PRODUCT_INFO_div2)
    RAW_PRODUCT_DESC2 = doc.xpath(XPATH_PRODUCT_DESC2)
    
    BRAND_img = None
    if len(RAW_BRAND_img) > 0:
        BRAND_img = ' '.join(RAW_BRAND_img[0].split('=')[-1].split('+'))
    
    PRODUCT_INFO_li = [x.strip() for x in RAW_PRODUCT_INFO_li if x is not None and x.strip() != '']
    PRODUCT_INFO_div2 = getProductInfo(RAW_PRODUCT_INFO_div2)
    PRODUCT_DESC2 = ' '.join([x.strip() for x in RAW_PRODUCT_DESC2 if x.strip() != ''])
    
    return {
            'BRAND' : BRAND_img,
            'PRODUCT_INFO' : dict(zip(PRODUCT_INFO_li[::2],PRODUCT_INFO_li[1::2])) if len(PRODUCT_INFO_li) > 0 else PRODUCT_INFO_div2,
            'PRODUCT_DESC' : PRODUCT_DESC2
           }

df = pd.read_json('reviews_Women.jl', lines=True)

# exclude duplicate entries by asin
df = df.groupby('asin').head(1)

df.head()

print df.columns
print df.shape

# check for captcha
captcha = [x.replace('.html', '') for x in os.listdir('./html') if x.endswith('.html') and os.stat('./html/'+ x).st_size ==6467]

len(captcha)

df.BRAND.describe()

# check for null or empty strings
df[(df.BRAND.isnull()) | (df.BRAND == '')].groupby('BRAND')['BRAND'].count()

m_brand = df[(df.BRAND.isnull()) | (df.BRAND == '')]
# fix only missing brands
df.ix[m_brand.index,['BRAND']] = m_brand.apply(lambda x: extract(x['asin'])['BRAND'], axis=1)

df[(df.BRAND.isnull()) | (df.BRAND == '')].shape

#count empty dictionaries
m_pi = df[df.PRODUCT_INFORMATION==dict()]
m_pi.shape

m_pi[['asin','PRODUCT_INFORMATION']].head()

df.ix[m_pi.index,['PRODUCT_INFORMATION']] = m_pi.apply(lambda x: extract(x['asin'])['PRODUCT_INFO'], axis=1)

df[df.PRODUCT_INFORMATION==dict()].shape

df[df.PRODUCT_INFORMATION==dict()].head()

m_pd = df[(df.PRODUCT_DESCRIPTION == '') | df.PRODUCT_DESCRIPTION.isnull()]
m_pd.shape

m_pd[['asin','PRODUCT_DESCRIPTION']].head()

df.ix[m_pd.index,['PRODUCT_DESCRIPTION']] = m_pd.apply(lambda x: extract(x['asin'])['PRODUCT_DESC'], axis=1)

df[(df.PRODUCT_DESCRIPTION == '') | df.PRODUCT_DESCRIPTION.isnull()].shape

df[(df.PRODUCT_DESCRIPTION == '') | df.PRODUCT_DESCRIPTION.isnull()].tail()

# save dataframe to file

df.to_hdf('scraped.hd5', key='womens', complib='blosc', complevel=9, mode='w')

def View(html):
    # this function injects html into the output cell to display html inline
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "html document", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + html + '\';'
    s += '</script>'

    return(HTML(s))

# source = u''.join([x.decode('utf-8') for x in open('./html/B0000862FI.html', 'rb')])
# View(source)

