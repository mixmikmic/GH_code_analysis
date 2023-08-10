from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
plt.style.use("ggplot")

url = "http://www.foodauthority.nsw.gov.au/penalty-notices/default.aspx?template=results"

r = requests.get(url)
soup = BeautifulSoup(r.text, "lxml")      
table = soup.find_all("table")

df = pd.DataFrame(columns = ['trade_name', 'suburb', 'council', 'penalty_no',
                             'date', 'party_served', 'desc'])

for t in table:
    table_body = t.find('tbody')
    try:
        rows = table_body.find_all('tr')
        for tr in rows:
            temp = []
            try:
                cols = tr.find_all('td')
                href = cols[3].find_all('a')
                temp.append(href[0].get('title'))
                for col in cols:
                    try:
                        temp.append(col.string.strip())
                    except:
                        pass
                
                df = df.append(pd.Series(temp, df.columns), ignore_index = True)
            
            except:
                print("Error reading row.")
                
        
    except:
        print("Error reading table body.")

df.head()

df.info()

df.columns = ["desc", "trade_name", "suburb", "council", "number", "date", "party_served"]
df.number = np.int64(df.number)
df.head()

url_2 = "http://www.foodauthority.nsw.gov.au/penalty-notices/default.aspx?template=detail&itemId="

df_2 = pd.DataFrame(columns = ["number", "trade_name", "address", "council",
                                "date_of_offence", "offence_code", "nature",
                                "penalty", "party_served", "date_served",
                                "issuer"])

for id_2 in df["penalty_id"]:
    temp_url = url_2 + id_2
    r = requests.get(temp_url)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find_all("table")
    
    for t in table:
        table_body = t.find('tbody')
        
        try:
            rows = table_body.find_all('tr')
            temp = []
            for tr in rows:
                try:
                    col = tr.find_all('td')
                    column_1 = col[1].string.strip()
                    temp.append(column_1)
                    
                except:
                    print("Error in row.")
        
        
        except:
            print("Error in table body.")
            
    df_2 = df_2.append(pd.Series(temp, df_2.columns), ignore_index = True)


df_2.head()

df_full = pd.merge(df, df_2, on = "number", how = "left", suffixes = ("_basic", "full"))
df_full = df_full.drop(["trade_namefull", "councilfull", "date_of_offence", "party_servedfull"], axis = 1)
df_full["penalty"] = df_full["penalty"].str.replace("$", "")
df_full["penalty"] = np.int64(df_full["penalty"].str.replace(",", ""))
df_full.head()

(df_full.groupby(["suburb"]).count()["desc"]/df.shape[0]).sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Proportion of Penalties")
plt.show()

df_full.groupby(["council_basic"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Number of Penalties")
plt.show()

df_full.groupby(["trade_name_basic"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Number of Penalties")
plt.show()

df_full.groupby(["offence_code"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind='barh')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel("Number of Penalties")
plt.show()

df_full.groupby(["nature"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind='barh')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel("Number of Penalties")
plt.ylabel("Nature of Offence")
plt.show()

df_full.groupby(["date"]).count()["penalty"].cumsum().plot()
plt.ylabel("Cumulative Number of Penalties")
plt.show()

df_full.groupby(["date"]).count()["penalty"].plot()
plt.ylabel("Number of Penalties")
plt.show()

df_full.groupby(["trade_name_basic"]).sum()["penalty"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Total Fines ($)")
plt.show()

temp = df_full.groupby(["trade_name_basic"]).sum()
temp["count"] = df_full.groupby(["trade_name_basic"]).count()["penalty"]

plt.scatter(temp["count"], temp["penalty"])
plt.xlabel("Number of Penalties")
plt.ylabel("Total Fines ($)")
plt.show()

df_full.groupby(["issuer"]).count()["penalty"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.ylabel("Number of Penalties")
plt.show()

from wordcloud import WordCloud, STOPWORDS

text = df_full["nature"].to_string()
text = str(text.split())

stopwords = set(STOPWORDS)
stopwords.add("to'")
stopwords.add("the'")

wordcloud = WordCloud(background_color = "white", max_words = 200, 
                      stopwords = stopwords, width = 2600, height = 1400).generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

