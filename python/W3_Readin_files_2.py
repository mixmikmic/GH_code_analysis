import pandas as pd

data_csv = pd.read_csv("titanic.csv")

data_csv.head()

data_txt = pd.read_csv("imagine_lyrics.txt", sep=" ")

data_txt.head()

data_html = pd.read_html("https://careercenter.am/")

data_html = pd.read_html("https://careercenter.am/ccidxann.php")

data_html.head()

print data_html

len(data_html)

data_html[0]

data_html[1]

data_html[2]

data_html[3]

data = data_html[0][1]

data.head()

data.to_csv("careercenter_data.csv")

