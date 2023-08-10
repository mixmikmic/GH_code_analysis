import pymysql.cursors
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime, timedelta

# Connect to the database
connection = pymysql.connect(host='192.168.0.30',
                             user='hass',
                             password='12345',
                             db='homeassistant',
                             charset='utf8',  # 'utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

df = pd.read_sql('SELECT * FROM homeassistant.states', con=connection)  # Takes 30 seconds to read

df.head()

for entity_id in df['entity_id'].unique():
    print(entity_id)

query = "SELECT entity_id, COUNT(*) FROM states GROUP BY entity_id ORDER by 2 DESC"
print("Performing a query: {}".format(query))

main_df = pd.read_sql(query, con=connection)  # Takes 30 seconds to read

main_df

main_df.columns = ['entity', 'Number of Changes']

# setting the entity name as an index of a new dataframe and sorting it \
# by the Number of Changes
ordered_indexed_df = main_df.set_index(['entity']).    sort_values(by='Number of Changes')

# displaying the data as a horizontal bar plot with a title and no legend
changesplot = ordered_indexed_df.plot(kind='barh', title='Number of Changes to Home Assistant per entity', figsize=(15, 10), legend=False)

# specifying labels for the X and Y axes
changesplot.set_xlabel('Number of Changes')
changesplot.set_ylabel('Entity name')

stmt = text("SELECT * FROM states where last_changed>=:date_filter")

# bind parameters to the stmt value, specifying the date_filter to be 10 days \
# before today
stmt = stmt.bindparams(date_filter=datetime.now()-timedelta(days=20))



