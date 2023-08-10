from SehirParser import *

import psycopg2

import ast

df = pd.read_csv("datasets/fb_users2.csv", index_col="Unnamed: 0")
df.head()

df.groupby("membership").count()

# df.head(10).to_csv("datasets/fb_users_toy.csv")

cols = {"id":"id", "name":"full_name", "screen_name":"full_name"}
fb_sp = SehirParser('datasets/contacts.csv', "datasets/fb_users2.csv", cols)

fb_sp.twitter_users_count

fb_df, sehir_matches_fb_df = fb_sp.get_sehir_matches_df()

sehir_matches_fb_df.head()

len(sehir_matches_fb_df)

fb_df["sehir_matches"] = fb_df["sehir_matches"].apply(lambda x: x[0][0])
fb_df = fb_df.rename(columns={"GUID":"fb_ID"}).set_index("fb_ID")
fb_df = fb_df[["full_name","sehir_matches","membership"]]
fb_df.head()

fb_df.to_csv('datasets/sehir_fb_matches2.csv', index_label="fb_ID")

len(fb_df)

fb_df.groupby("sehir_matches").count().shape

twitter_matches_by_guid = pd.read_csv('datasets/sehir_matches.csv', index_col="GUID.1").drop("GUID", axis=1)
twitter_matches_by_guid["sehir_matches"] = twitter_matches_by_guid["sehir_matches"].apply(lambda x: ast.literal_eval(x)[0][0])
twitter_matches_by_guid = twitter_matches_by_guid.drop("twitter_name",axis=1).reset_index()                                .rename(columns={
                                    "cleaned_twitter_name":"twitter_name",
                                    "GUID.1":"twitter_ID"}).set_index("twitter_ID")
twitter_matches_by_guid.head()

twitter_matches_by_guid.groupby("sehir_matches").count().shape

twitter_fb = twitter_matches_by_guid.reset_index().merge(fb_df.reset_index(), left_on="sehir_matches", right_on="sehir_matches")
twitter_fb.head()

twitter_fb.set_index("sehir_matches").loc["muhammed caki"]

len(fb_df)

len(twitter_fb)

twitter_fb.groupby("sehir_matches").count().shape

twitter_fb.to_csv('datasets/twitter_fb_matches.csv', index_label="ID")



