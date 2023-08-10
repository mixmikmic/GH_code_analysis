# Import Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

stories_df = pd.read_csv("Data/stories_for_viz.csv")
stories_df.head()

stories_df.count()

stories_df["categoryName"].value_counts()

stories_df["languageName"].value_counts()

font={'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

category_grp = stories_df.groupby("categoryName", as_index=False)
stories_by_category = category_grp.count()
stories_by_category

# create a bar plot
plt.figure(figsize=(10,8))

sns.barplot(x=stories_by_category["categoryName"], y=stories_by_category["id"], data=stories_by_category, palette=sns.hls_palette(10, l=.4, s=.3), label="Stories by Category")

# set plot properties
plt.title("Distribution of stories by Category", fontdict=font)
plt.ylabel("Number of stories", fontdict=font)
plt.xlabel("Category", fontdict=font)
plt.xticks(rotation="vertical")
plt.tight_layout()
# plot and save
plt.savefig("Images/dist_category_bar.png")
plt.show()

# create pie chart
plt.figure(1, figsize=(10,8 ))

patches, texts, autotexts = plt.pie(stories_by_category["id"], 
                                    labels=stories_by_category["categoryName"], 
                                    colors=sns.hls_palette(10, l=.4, s=.4),
                                    autopct='%1.1f%%', startangle=140)

# set plot properties
plt.legend(patches, stories_by_category["categoryName"], loc="best", bbox_to_anchor=(1.2,1))
plt.axis('equal')
plt.title("Distribution of stories by Categories\n", fontdict=font)
plt.tight_layout()
# plot and save
plt.savefig("Images/dist_category_pie.png")
plt.show()

language_grp = stories_df.groupby("languageName", as_index=False)
stories_by_language = language_grp.count()
stories_by_language

# create bar chart
sns.barplot(x=stories_by_language["languageName"], y=stories_by_language["id"], 
            data=stories_by_language, palette=sns.hls_palette(10, l=.4, s=.3), label="Stories by Language")

# set plot properties
plt.title("Distribution of stories by Language", fontdict=font)
plt.ylabel("Number of stories", fontdict=font)
plt.xlabel("Language", fontdict=font)
plt.tight_layout()

# plot and save image
plt.savefig("Images/dist_language_bar.png")
plt.show()

# create a pie chart
plt.figure(1, figsize=(10,8 ))

patches, texts, autotexts = plt.pie(stories_by_language["id"], 
                                    labels=stories_by_language["languageName"],
                                    colors=sns.hls_palette(10, l=.4, s=.4),
                                    autopct='%1.1f%%', startangle=140)

# set pie chart properties
plt.legend(patches, stories_by_language["languageName"], loc="best", bbox_to_anchor=(1,0))
plt.axis('equal')
plt.title("Distribution of stories by Language", fontdict=font)
plt.tight_layout()

# plot and save image
plt.savefig("Images/dist_language_pie.png")
plt.show()


popularity_df = category_grp.mean()
popularity_df

ax = sns.barplot(x=popularity_df["categoryName"], y=popularity_df["readCount"], data=popularity_df, 
                 palette=sns.hls_palette(10, l=.4, s=.3), label="Popularity by Stories read")
plt.title("Popularity by stories read", fontdict=font)
plt.ylabel("Number of stories read", fontdict=font)
plt.xlabel("Category", fontdict=font)
plt.xticks(rotation="vertical")
plt.tight_layout()
plt.savefig("Images/popularity_category_by_readCount.png")
plt.show()

sns.barplot(x=popularity_df["categoryName"], y=popularity_df["commentCount"], data=popularity_df, 
            palette=sns.hls_palette(10, l=.4, s=.3), label="Popularity by Comments")
plt.title("Popularity by Comments", fontdict=font)
plt.ylabel("Number of Comments", fontdict=font)
plt.xlabel("Category", fontdict=font)
plt.xticks(rotation="vertical")
plt.tight_layout()
plt.savefig("Images/popularity__category_by_comments.png")
plt.show()

sns.barplot(x=popularity_df["categoryName"], y=popularity_df["voteCount"], data=popularity_df, 
            palette=sns.hls_palette(10, l=.4, s=.3), label="Popularity by Votes")
plt.title("Popularity by Votes", fontdict=font)
plt.ylabel("Number of Votes", fontdict=font)
plt.xlabel("Category", fontdict=font)
plt.xticks(rotation="vertical")
plt.tight_layout()
plt.savefig("Images/popularity_category_by_votes.png")
plt.show()

stories_df.info()

# Function to get the tags as a list from the tags columns
def get_tag_list(tag_str):
    tag_str = tag_str.strip("['")
    tag_str = tag_str.strip("']")
    tag_str = tag_str.strip("', '")
    tags = tag_str.split("', '")
    return tags

# get all the tags from the stories and get the top 10 popular tags
all_tags = []
for index,row in stories_df.iterrows():
    tag = row["tags"]
    all_tags += get_tag_list(tag)

tags_df = pd.DataFrame({"tag": all_tags})
tags_counts = pd.DataFrame(tags_df["tag"].value_counts())
popular_tags = tags_counts.iloc[0:10, :]

popular_tags.reset_index(inplace=True)
popular_tags = popular_tags.rename(columns={"tag":"count","index":"tag"})
popular_tags

# sns.barplot(x="tag", y="count", data=popular_tags, palette='rainbow')
# plt.title("Story count by tags", size=15)
# plt.ylabel("Number of stories", size=10)
# plt.xlabel("Tags", size=10)
# plt.xticks(rotation="vertical")
# plt.tight_layout()
# plt.savefig("Images/tag_distribution_bar.png")
# plt.show()



plt.figure(figsize=(10,8))
sns.barplot(x="tag", y="count", data=popular_tags, palette=sns.hls_palette(10, l=.4, s=.3))
plt.title("Story count by tags",fontdict=font)
plt.ylabel("Number of stories",fontdict=font)
plt.xlabel("Tags",fontdict=font)
plt.xticks(rotation="vertical")
plt.savefig("Images/tag_distribution_bar.png")
plt.show()

# # create a pie chart
# plt.figure(1, figsize=(6,6))
# patches, texts, autotexts = plt.pie(popular_tags["count"], 
#                                     labels=popular_tags["tag"],
#                                     autopct='%1.1f%%', startangle=140)

# # set pie chart properties
# #plt.legend(patches, popular_tags["tag"], loc="best", bbox_to_anchor=(1.2,1))
# plt.legend(patches, popular_tags["tag"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# plt.axis('equal')
# plt.title("Distribution of stories by tags\n", size=15)
# plt.tight_layout()

# # plot and save image
# plt.savefig("Images/tag_distribution_pie.png")
# plt.show()


# create a pie chart
plt.figure(1, figsize=(10,8 ))
patches, texts, autotexts = plt.pie(popular_tags["count"], 
                                    labels=popular_tags["tag"],colors=sns.hls_palette(10, l=.4, s=.4),
                                    autopct='%1.1f%%', shadow=False, startangle=140)

# set pie chart properties
#plt.legend(patches, popular_tags["tag"], loc="best", bbox_to_anchor=(1.2,1))
plt.axis('equal')
plt.title("Distribution of stories by tags\n",fontdict=font)

# plot and save image
plt.savefig("Images/tag_distribution_pie.png")
plt.show()

# Getting the vote count, comment count and read count by tags
tags_dict = {}
read_dict = {}
comment_dict = {}
vote_dict = {}

# Loop through the stories and add the counts for each tag of the story
for index,row in stories_df.iterrows():
    story_tags = row["tags"]
    read_count = row["readCount"]
    comment_count = row["commentCount"]
    vote_count = row["voteCount"]
    
    # For each story, loop through the tags and see which one matches the tags list.
    for tag_index, tag_row in popular_tags.iterrows():
        tag_name = tag_row["tag"]
        if tag_name in get_tag_list(story_tags):
            if tag_name in tags_dict.keys():
                tags_dict[tag_name] += 1
                read_dict[tag_name] += read_count
                comment_dict[tag_name] += comment_count
                vote_dict[tag_name] += vote_count
            else:
                tags_dict[tag_name] = 1
                read_dict[tag_name] = read_count
                comment_dict[tag_name] = comment_count
                vote_dict[tag_name] = vote_count
    
# create data frame for all types of counts
# vote counts
vote_df = pd.DataFrame({"tag": list(vote_dict.keys()),
                       "votes": list(vote_dict.values()) })

# comment counts
comment_df = pd.DataFrame({"tag": list(comment_dict.keys()),
                       "comments": list(comment_dict.values())})

# comment counts
reads_df = pd.DataFrame({"tag": list(read_dict.keys()),
                       "reads": list(read_dict.values())})


# sns.barplot(x="tag", y="votes", data=vote_df, palette='rainbow')
# plt.title("Popularity by tags", size=15)
# plt.ylabel("Number of votes", size=10)
# plt.xlabel("Tags", size=10)
# plt.xticks(rotation="vertical")
# plt.tight_layout()
# plt.savefig("Images/popularity_tags_by_votes.png")
# plt.show()


plt.figure(figsize=(10,8))

sns.barplot(x="tag", y="votes", data=vote_df, palette=sns.hls_palette(10, l=.4, s=.3))
plt.title("Popularity by tags",fontdict=font)
plt.ylabel("Number of votes",fontdict=font)
plt.xlabel("Tags",fontdict=font)
plt.xticks(rotation="vertical")
plt.savefig("Images/popularity_tags_by_votes.png")
plt.show()

# sns.barplot(x="tag", y="comments", data=comment_df, palette='rainbow')
# plt.title("Popularity by tags", size=15)
# plt.ylabel("Number of comments", size=10)
# plt.xlabel("Tags", size=10)
# plt.xticks(rotation="vertical")
# plt.tight_layout()
# plt.savefig("Images/popularity_tags_by_comments.png")
# plt.show()

plt.figure(figsize=(10,8))

sns.barplot(x="tag", y="comments", data=comment_df, palette=sns.hls_palette(10, l=.4, s=.3))
plt.title("Popularity by tags",fontdict=font)
plt.ylabel("Number of comments",fontdict=font)
plt.xlabel("Tags",fontdict=font)
plt.xticks(rotation="vertical")
plt.savefig("Images/popularity_tags_by_comments.png")
plt.show()

# sns.barplot(x="tag", y="reads", data=reads_df, palette='rainbow')
# plt.title("Popularity by tags", size=15)
# plt.ylabel("Number of comments", size=10)
# plt.xlabel("Tags", size=10)
# plt.xticks(rotation="vertical")
# plt.tight_layout()
# plt.savefig("Images/popularity_tags_by_reads.png")
# plt.show()


plt.figure(figsize=(10,8))

sns.barplot(x="tag", y="reads", data=reads_df, palette=sns.hls_palette(10, l=.4, s=.3))
plt.title("Popularity by tags",fontdict=font)
plt.ylabel("Number of comments",fontdict=font)
plt.xlabel("Tags",fontdict=font)
plt.xticks(rotation="vertical")
plt.savefig("Images/popularity_tags_by_reads.png")
plt.show()





