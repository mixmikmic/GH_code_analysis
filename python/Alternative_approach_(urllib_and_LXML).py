import urllib2
from lxml import html

url = "https://careercenter.am/ccidxann.php"

response = urllib2.urlopen(url)
page = response.read()

tree = html.document_fromstring(page)

tables = tree.cssselect("table")

len(tables)

tables[-1].text_content()

our_table = tree.cssselect('[width="100%"],[border="0"]')

for i in our_table:
    print(i.text_content())

tree.xpath('//table')[-1].text_content()

tree.xpath('//table[@border="0"]')[-1].text_content()

tree.xpath('//table/@border')

