import pandas

url = "http://craftcans.com/db.php?search=all&sort=beerid&ord=desc&view=text"

scraped_data = pandas.read_html(url)

len(scraped_data)

data = scraped_data[-1]

data.head()

data.to_excel("craftcans.xlsx")

