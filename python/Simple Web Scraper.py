from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

df=pd.DataFrame()
df=pd.DataFrame(columns=['emails'])
for x in range(0,10):
    url='http://help.websiteos.com/websiteos/example_of_a_simple_html_page.htm'
    html=urlopen(url)
    soup=BeautifulSoup(html.read(),'lxml')
    subSoup=soup.find_all('p',class_='whs2');
    for elm in subSoup:
        ret=elm.contents[0]
        if '<a href' in ret and '@' in ret:
            email=ret[25+len('mailto')+1:-2]
    df.loc[x,'emails']='support2@yourcompany%s.com' %(x)
df

#df.to_csv('Scraped_emails.csv',index=False)

