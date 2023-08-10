get_ipython().magic('matplotlib inline')
# Libraries

from pandas import read_csv, value_counts
import numpy as np
import matplotlib.pyplot as plt

#read in data
file = "../../data/formatted_data/2016espnprojstats.csv"
raw_data = read_csv(file, na_values = ["","--"])

#only take the top 300
data = raw_data.ix[:300,:]

#Create array of last names
last_names = data["Name"].copy()
for index,name in last_names.iteritems():
    result = name
    if name.find("D/ST") < 0:
        arr = name.split(" ")[1:]
        result = " ".join(arr)
    last_names.set_value(index, result)

#show relative values for each column
data.drop("Rank",1).describe()

# Counts at each position
position_group = data.groupby('Position')
position_group.mean()#.size()

#Create color mappings array so that each position has a different color
colors = {"RB": "red",
          "WR": "blue",
          "QB": "green",
          "TE": "black",
          "K": "yellow",
          "D/ST": "violet"}

pos = data["Position"].tolist()
for index, item in enumerate(pos):
    pos[index] = colors[item]

fig = plt.figure()
ax = fig.add_subplot("111")

ax.set_title("Points vs. Rank")
ax.set_ylabel("Points")
ax.set_xlabel("Rank")

ax.scatter(data["Rank"], data["PTS"], marker = "x", c = pos)

ax.plot(data["Rank"], np.poly1d(np.polyfit(data["Rank"], data["PTS"], 1))(data["Rank"]), c = "orange")

plt.show()

# Filter QB's
qb_data = data[data["Position"] == "QB"]

qb_data = qb_data.drop(["Position", "REC", "RECYDS", "RECTD"], 1)

#qb averages 
qb_data.describe()

# qb graph
qb_fig = plt.figure(figsize=(16, 6))
qb_ax = qb_fig.add_subplot("111")

qb_ax.set_title("Quarterback Points vs. Rank")
qb_ax.set_ylabel("Points")
qb_ax.set_xlabel("Rank")
qb_ax.set_xlim(0,300)

qb_ax.scatter(qb_data["Rank"], qb_data["PTS"], marker = "x", c = colors["QB"])

qb_ax.plot(qb_data["Rank"], np.poly1d(np.polyfit(qb_data["Rank"], qb_data["PTS"], 1))(qb_data["Rank"]), c = "orange")

for index, name in enumerate(qb_data["Name"]):
    #data point
    rank = qb_data["Rank"].iloc[index]
    pts = qb_data["PTS"].iloc[index]
    
    #get last name of player
    name = last_names[rank-1]
    
    #add label to the data point
    qb_ax.text(rank+1,pts+1, name, fontsize=8)

plt.show()

# Filter RB's
rb_data = data[data["Position"] == "RB"]

rb_data = rb_data.drop(["Position", "C", "A", "PASSTD"], 1)

#rb averages 
rb_data.describe()

# rb graph
rb_fig = plt.figure(figsize=(16, 10))
rb_ax = rb_fig.add_subplot("111")

rb_ax.set_title("Runningback Points vs. Rank")
rb_ax.set_ylabel("Points")
rb_ax.set_xlabel("Rank")
rb_ax.set_xlim(0,300)
rb_ax.set_ylim(0,235)

rb_ax.scatter(rb_data["Rank"], rb_data["PTS"], marker = "x", c = colors["RB"])

rb_ax.plot(rb_data["Rank"], np.poly1d(np.polyfit(rb_data["Rank"], rb_data["PTS"], 1))(rb_data["Rank"]), c = "orange")

for index, name in enumerate(rb_data["Name"]):
    #data point
    rank = rb_data["Rank"].iloc[index]
    pts = rb_data["PTS"].iloc[index]
    
    #get last name of player
    name = last_names[rank-1]
    
    #add label to the data point
    rb_ax.text(rank+1,pts+1, name, fontsize=8)

plt.show()

# Filter WR's
wr_data = data[data["Position"] == "WR"]

wr_data = wr_data.drop(["Position", "C", "A", "PASSTD"], 1)

#rb averages 
wr_data.describe()

# wr graph
wr_fig = plt.figure(figsize=(16, 12))
wr_ax = wr_fig.add_subplot("111")

wr_ax.set_title("Wide Receiver Points vs. Rank")
wr_ax.set_ylabel("Points")
wr_ax.set_xlabel("Rank")
wr_ax.set_ylim(10,250)
wr_ax.set_xlim(-5,300)

wr_ax.scatter(wr_data["Rank"], wr_data["PTS"], marker = "x", c = colors["WR"])

wr_ax.plot(wr_data["Rank"], np.poly1d(np.polyfit(wr_data["Rank"], wr_data["PTS"], 1))(wr_data["Rank"]), c = "orange")

for index, name in enumerate(wr_data["Name"]):
    #data point
    rank = wr_data["Rank"].iloc[index]
    pts = wr_data["PTS"].iloc[index]
    
    #get last name of player
    name = last_names[rank-1]
    
    #add label to the data point
    wr_ax.text(rank+1,pts+1, name, fontsize=8)

plt.show()

# Filter TE's
te_data = data[data["Position"] == "TE"]

te_data = te_data.drop(["Position", "C", "A", "PASSTD"], 1)

#rb averages 
te_data.describe()

# te graph
te_fig = plt.figure(figsize=(16, 6))
te_ax = te_fig.add_subplot("111")

te_ax.set_title("Tightend Points vs. Rank")
te_ax.set_ylabel("Points")
te_ax.set_xlabel("Rank")
te_ax.set_xlim(0,300)

te_ax.scatter(te_data["Rank"], te_data["PTS"], marker = "x", c = colors["TE"])

te_ax.plot(te_data["Rank"], np.poly1d(np.polyfit(te_data["Rank"], te_data["PTS"], 1))(te_data["Rank"]), c = "orange")

for index, name in enumerate(te_data["Name"]):
    #data point
    rank = te_data["Rank"].iloc[index]
    pts = te_data["PTS"].iloc[index]
    
    #get last name of player
    name = last_names[rank-1]
    
    #add label to the data point
    te_ax.text(rank+1,pts+1, name, fontsize=10)

plt.show()

# Filter Flex

# Filter K's

# Filter D/ST's

