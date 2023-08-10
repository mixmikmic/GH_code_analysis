import requests
from scrapy.http import TextResponse

url = "http://quotes.toscrape.com/page/1/"

r = requests.get(url)
response = TextResponse(r.url, body=r.text, encoding='utf-8')

response.css('title')

response.css('title').extract()

response.css('title::text').extract()

type(response.css('title::text').extract())

response.css('title::text').extract_first()

type(response.css('title::text').extract_first())

response.css('h1').extract()

response.css('h1::text').extract()

response.css('h1 a').extract()

response.css('a[style="text-decoration: none"]').extract()

response.css('a[style="text-decoration: none"]::text').extract()

response.css('a[style="text-decoration: none"]::attr(href)').extract()

# expression explanation: find Quotes, a whitespace, anything else
# return only anything else component
response.css('h1 a::text').re('Quotes\s(.*)')

response.css('h1 a::text').re('(\S+)\s(\S+)\s(\S+)')

response.xpath('//title').extract()

response.xpath('//title/text()').extract()

response.xpath('//h1/a').extract()

response.xpath('//h1/a/text()').extract()

response.xpath('//h1/a/@href').extract()

