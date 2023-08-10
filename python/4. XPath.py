import requests
from lxml import html

get_ipython().run_cell_magic('HTML', '', '<html>\n  <body>\n    <h1>Favorite Python Librarires</h1>\n    <ul>\n      <li>Numpy</li>\n      <li>Pandas</li>\n      <li>requests</li>\n    </ul>\n  </body>\n</html>')

html_code = In[2]
html_code = html_code[42:-2].replace("\\n","\n")
print(html_code)

doc = html.fromstring(html_code)

title = doc.xpath("/html/body/h1")[0]
title

title.text

title = doc.xpath("/html/body/h1/text()")[0]
title

item_list = doc.xpath("/html/body/ul/li")
item_list

doc = html.fromstring(html_code)
item_list = doc.xpath("/html/body/ul/li/text()")
item_list

doc = html.fromstring(html_code)
item_list = doc.xpath("//li/text()")
item_list

doc = html.fromstring(html_code)
item_list = doc.xpath("/html/body/ul/li[1]/text()")
item_list

get_ipython().run_cell_magic('HTML', '', '<html>\n  <body>\n    <h1 class="text-muted">Favorite Python Librarires</h1>\n    <ul class="nav nav-pills nav-stacked">\n      <li role="presentation"><a href="http://www.numpy.org/">Numpy</a></li>\n      <li role="presentation"><a href="http://pandas.pydata.org/">Pandas</a></li>\n      <li role="presentation"><a href="http://python-requests.org/">requests</a></li>\n    </ul>\n    <h1 class="text-success">Favorite JS Librarires</h1>\n    <ul class="nav nav-tabs">\n      <li role="presentation"><a href="http://getbootstrap.com/">Bootstrap</a></li>\n      <li role="presentation"><a href="https://jquery.com/">jQuery</a></li>\n      <li role="presentation"><a href="http://d3js.org/">d3.js</a></li>\n    </ul>\n</html>')

html_code = In[11]
html_code = html_code[42:-2].replace("\\n","\n")
print(html_code)

doc = html.fromstring(html_code)

title = doc.xpath("/html/body/h1[@class='text-muted']/text()")[0]
title

item_list = doc.xpath("/html/body/ul[contains(@class,'nav-stacked')]/li/a/text()")
item_list

item_list = doc.xpath("/html/body/ul[contains(@class,'nav-stacked')]/li/a/@href")
item_list

response = requests.get("http://www.wikipedia.org")
doc = html.fromstring(response.content, parser=html.HTMLParser(encoding="utf-8"))

lang_list = doc.xpath("//div[@class='langlist langlist-large hlist'][1]/ul/li/a/text()")
lang_list



