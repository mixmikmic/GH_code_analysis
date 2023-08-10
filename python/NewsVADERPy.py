# ----------------------------------------------------------------------
# Step 1: Import necessary modules and environment
# ----------------------------------------------------------------------

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# This file contains all Twitter-related actions, so no need to import here
import tweetParser as tp

# ----------------------------------------------------------------------
# Step 2: Call API, get tweets, and parse tweets into a dataframe+CSV
# ----------------------------------------------------------------------

# create list of target news organizations' Twitter handles
targetNewsOrg_list = ["BBC", "FoxNews", "nytimes", "BreitbartNews",
                      "CBSNews", "USATODAY"]

# create and set color palette for all charts
orgPalette = sns.color_palette("bright", len(targetNewsOrg_list))
sns.set_palette(orgPalette)

# define number of tweets we want to pull from each org
numTweets = 150

# break into increments of 10
numCycles = int(round(numTweets/10))

# create dict to store dictionaries generated during analysis
completeResults_df = tp.parseTweets(targetNewsOrg_list, numCycles)

# rearrange columns to be more sensible
completeResults_df = completeResults_df[["handle", "count", "compound",
                                         "positive", "negative", "neutral",
                                         "date", "text"]]
completeResults_df.to_csv("TweetsAnalyzed.csv")

completeResults_df.head()

# ----------------------------------------------------------------------
# Step 3: Generate first plot: scatterplot of last 100 tweets showing 
# compound sentiment and sorted by relative timestamp
# ----------------------------------------------------------------------
#set style to be seaborn
sns.set()

# generate overall plot
compoundSentByTime_plot = sns.lmplot(x="count", y="compound", 
                                data=completeResults_df, 
                                palette=orgPalette, hue='handle',
                                fit_reg=False, legend=True, size=8, 
                                scatter_kws={'s':175, 'alpha':0.75,
                                             'edgecolors':'face', 
                                             'linewidths':2})
plt.xlabel("Tweet Count (From Newest to Oldest)",size=14)
plt.ylabel("Compound Sentiment Score", size=14)
plt.title("Tweet Compound Sentiment Analysis Over Time by News Organization", 
          size=16)
plt.locator_params(nbins=5)
plt.xticks(rotation=45)
plt.savefig("CompoundSentimentAnalysis_Scatterplot.png")

#generate subplots
orgPalette = sns.color_palette("bright", len(targetNewsOrg_list))
compoundSentByTime_subplots = sns.lmplot(x="count", y="compound", 
                                data=completeResults_df, 
                                col="handle", col_wrap = 2,
                                palette=orgPalette, hue='handle',
                                fit_reg=False, legend=True, size=8,
                                scatter_kws={'s':175, 'alpha':0.5,
                                             'edgecolors':'face', 
                                             'linewidths':1})
plt.savefig("CompoundSentimentAnalysis_Subplots.png")


plt.locator_params(nbins=5)
plt.xticks(rotation=45)
plt.show(compoundSentByTime_plot)
plt.show(compoundSentByTime_subplots)

# ----------------------------------------------------------------------
# Step 4: Generate second plot: bar plot showing overall compound 
# sentiment in the last 100 tweets
# ----------------------------------------------------------------------

# generate dataframe
meanCompoundSent_df = pd.DataFrame(completeResults_df.groupby("handle").mean()["compound"])
meanCompoundSent_df.reset_index(level=0, inplace=True)

# generate x + y
x_axis = np.arange(len(targetNewsOrg_list))
y_axis = meanCompoundSent_df["compound"]

# create bar plot
plt.bar(x_axis, y_axis, color=orgPalette, width=1, align='edge', 
        alpha=0.75, linewidth=1, edgecolor='black')
tick_locations = [value + 0.4 for value in x_axis]
plt.xticks(tick_locations, meanCompoundSent_df['handle'], rotation=35)
plt.xlabel('News Organization', size=14)
plt.ylabel('Mean Compound Sentiment', size=14)
plt.ylim(-0.15, 0.17)
plt.title('Average Compound Sentiment by News Organization', size=16)
plt.savefig("avgCompoundSentimentBarchart.png")
plt.show()

