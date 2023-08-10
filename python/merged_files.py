import glob
import pandas as pd
import os

path = r'raw_data'
all_files = glob.glob(os.path.join(path, "*.csv"))

df_from_each_file = (pd.read_csv(x) for x in all_files)
concatenated_df = pd. concat(df_from_each_file, ignore_index=True)

concatenated_df.head()

concatenated_df.tail()

concatenated_df.describe()

concatenated_df.count()

#concatenated_df.to_csv("last_year.csv", index=False)

Summer_df=concatenated_df[(concatenated_df['starttime'] > '2017-06-01') & (concatenated_df['starttime'] <= '2017-08-31')]
Summer_df.head()

Summer_df.tail()

Summer_df.to_csv("summer_months.csv", index=False)

Winter_df=concatenated_df[(concatenated_df['starttime'] > '2017-12-01') & (concatenated_df['starttime'] <= '2018-02-30')]
Winter_df.head()

Winter_df.tail()

print(Summer_df.count())
#Summer_df.dtypes

#Check for repeats -NO DUPLICATES FOUND
#Summer_df.drop_duplicates()
#print(Summer_df.count())
#print(Winter_df.count())
#Winter_df.drop_duplicates()
#print(Winter_df.count())

Winter_df.to_csv("winter_months.csv", index=False)



