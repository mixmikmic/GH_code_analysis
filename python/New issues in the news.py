import eikon as tr
import pandas as pd

tr.set_app_id('your_app_id')

from datetime import date

start_date, end_date = date(2016, 1, 1), date.today()
q = "Product:IFREM AND Topic:ISU AND Topic:EUB AND (\"PRICED\" OR \"DEAL\")"
headlines = tr.get_news_headlines(query=q, date_from=start_date, date_to=end_date, count=100)
headlines.head()

from IPython.core.display import HTML
html = tr.get_news_story('urn:newsml:reuters.com:20170405:nIFR5LpzRX:1')
HTML(html)

from lxml import html
import re

def termsheet_to_dict(storyId):
    x = tr.get_news_story(storyId)
    story = html.document_fromstring(x).text_content()
    matches = dict(re.findall(pattern=r"\[(.*?)\]:\s?([A-Z,a-z,0-9,\-,\(,\),\+,/,\n,\r,\.,%,\&,>, ]+)", string=story))
    clean_matches = {key.strip(): item.strip() for key, item in matches.items()}
    return clean_matches

termsheet_to_dict('urn:newsml:reuters.com:20170323:nIFR9z7ZFL:1')['NOTES']

result = []

index = pd.DataFrame(headlines, columns=['storyId']).values.tolist()

for i, storyId in enumerate(index):
    x = termsheet_to_dict(storyId[0])
    if x:
        result.append(x)

df = pd.DataFrame(result)
df.head()

df['Asset Type'].value_counts()

df[df['Country']=='RUSSIA']['Asset Type'].value_counts()

