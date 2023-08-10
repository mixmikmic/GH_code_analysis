# For users: Change the filenames as you like.

INPUTFILE = "POE_json2.json"
OUTPUTFILE = "results.csv"

# header
get_ipython().system('echo "[]" | jq -r \'["tweet_created_at","userID", "screen_name", "user_created_at"] | @csv\' > "csvdata.csv"')
get_ipython().system('cat $INPUTFILE | jq -r \'[(.created_at | strptime("%A %B %d %T %z %Y") | todate), .user.id_str, .user.screen_name, (.user.created_at | strptime("%A %B %d %T %z %Y") | todate)] | @csv\' >> "csvdata.csv"')
get_ipython().system('head -5 "csvdata.csv"')

import pandas as pd              

data = pd.read_csv("csvdata.csv", encoding = 'ISO-8859-1')
data2 = data.groupby(['userID', 'screen_name', 'user_created_at']).tweet_created_at.agg(['min', 'max'])
data3 = data2.reset_index()
data3.rename(columns={'min': 'first_tweet_date', 'max': 'last_tweet_date'}, inplace=True)
data3.head(5)

# the number of unique users
len(data3)

# Export the results to a csv file whose filename is OUTPUTFILE set by user in the beginning of thie notebook.
data3.to_csv(OUTPUTFILE, index=False)

