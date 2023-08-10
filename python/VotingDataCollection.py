# import the necessary libraries
from bs4 import BeautifulSoup
import pandas as pd

# create handle for BeautifulSoup instance
soup = BeautifulSoup(open("./Data/CQ Voting and Elections Collection.html"), "html.parser")

table = soup.findAll('td')

#take a look at what we've extracted

table

#pull out just the first row to see the headers

soup.findAll('tr')[1].findAll('td')

#but the second row of headers are what we actually care about. 

soup.findAll('tr')[2].findAll('td')

#extract just the text, and store each label in a list called column_headers
column_headers = [td.getText() for td in 
                  soup.findAll('tr')[2].findAll('td')]

#take a look
column_headers

column_headers[4] = "Republican Vote"
column_headers[5] = "Republican Candidate"
column_headers[6] = "Democratic Vote"
column_headers[7] = "Democratic Candidate"
column_headers[8] = "Highest Other Vote"
column_headers[9] = "Highest Other Candidate"
column_headers[12] = "Total Vote Percent Rep"
column_headers[13] = "Total Vote Percent Dem"
column_headers[14] = "Total Vote Percent Highest Other"
column_headers[15] = "Total Vote Percent Other"
column_headers[16] = "Major Party Vote Percent Rep"
column_headers[17] = "Major Party Vote Percent Dem"

column_headers

#subset the table to just include the data (rows three onwards), and store in data_rows
data_rows = soup.findAll('tr')[3:]

#take a look
data_rows

type(data_rows)

#extract only the data from the table sub-set, and store in race_data
race_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]

#take a look
race_data

#combine the headers with the data in one dataframe
df = pd.DataFrame(race_data, columns=column_headers)

column_headers.insert(12, "WinningParty")

column_headers

df = pd.DataFrame(race_data, columns=column_headers)

df

df = df[:-1]

df

#add a state column, fill in with AL
df.insert(2 , "State", "AL", allow_duplicates=False)

df

#remove district
df['CD'] = df['CD'].str.lstrip('District')
df

#remove "[*] next to candidate names"
df['Republican Candidate'] = df['Republican Candidate'].str.rstrip('[*]')
df['Democratic Candidate'] = df['Democratic Candidate'].str.rstrip('[*]')
df['Highest Other Candidate'] = df['Highest Other Candidate'].str.rstrip('[*]')

df

#drop the Total Vote Percent Other column
df.drop("Total Vote Percent Other", 1)

#export into a csv

df.to_csv("Alabama.csv", sep=',')

