import pandas as pd             # Bring in the Pandas Package
import numpy as np              # Numberical Python, we will use this at the end.
import matplotlib as mpl               # graphics package
import matplotlib.pyplot as plt # Our plotting tool

url = "https://github.com/mwaugh0328/Data_Bootcamp_Fall_2017/raw/master/data_bootcamp_1030/pwt_inclass.csv"                   # Here is the correct link
pwt = pd.read_csv(url, encoding = "latin1")  

# If this does not work, read this one in
# https://github.com/mwaugh0328/Data_Bootcamp_Fall_2017/raw/master/data_bootcamp_1030/pwt_inclass.csv

pwt.head()

pwt["gdp_pop"] = pwt["rgdpe"] / pwt["pop"] # This makes GDP per person or ``average income in that country''

pwt_2005 = pwt.set_index("year").loc[2005]        # This will get me just the 2005 year...

pwt_2005.head()

fig, ax = plt.subplots()

ax.hist(pwt_2005.gdp_pop, rwidth= 0.9, alpha = 0.75)

# The rwidth is the spacing between the bars...
#

ax.set_xlabel("PPP 2005 Dollars")

ax.set_title("Global Income Inequality\n")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# All the stuff is the same as before... below is the new stuff.

avg = pwt_2005.gdp_pop.mean() # This computes the average value.

ax.axvline(x=avg,           # Set the value equall to the average
           color='r',       # make the color red
           label='Average', # this is the label (shows up in the legend)
           linestyle='-',   # the line style
           linewidth=2)     # thickness of the line

message = "World Average \n" + str(round(avg,-1)) # Create the message, convert the number to a string,
                                                  # then add it 
                                                  
ax.text(avg + 1000, # This is the placement on the x-axis, I'm shifting it righ to see it better
        100, # placement on the y-axis
        message,  # The message
        horizontalalignment='left') # then alling everything on the left.

usval = float(pwt_2005[pwt_2005.countrycode == "USA"].gdp_pop)
# Note how I'm doing this, I'm slicing the data frame by a boolean operation. So take the value
# where the countrycode is USA, return gdp_pop. The one issue here is that it returns a dataframe
# not a floating point value. So I use float to convert it.

ax.axvline(x= usval, 
           color='k', 
           label= "USA", 
           linestyle='--', 
           linewidth=2) # Put in the US Value
# Same deal as above

message = "US Value \n" + str(round(usval,-1))

ax.text(usval +1000, 100, message, horizontalalignment='left')
# Same deal as above.

plt.savefig("hist.png", bbox_inches="tight", dpi = 600)

plt.show()

fix, ax = plt.subplots()

ax.scatter(pwt_2005["gdp_pop"], pwt_2005["hc"],     # x,y variables 
            alpha= 0.50) # Then this last command specifies how dark or light the bubbles are...

plt.show()

fig, ax = plt.subplots()

ax.scatter(np.log(pwt_2005["gdp_pop"]), np.log(pwt_2005["hc"]), # np.log() is taking a natural log transformation...
            alpha= 0.50) # Then this last command specifies how dark or light the bubbles are...

plt.show()

fig, ax = plt.subplots()

ax.scatter(np.log(pwt_2005["gdp_pop"]), np.log(pwt_2005["hc"]), # np.log() is taking a natural log transformation...
            s=pwt_2005['pop'], # THE NEW PART HERE! 
            alpha= 0.50) # Then this last command specifies how dark or light the bubbles are...


ax.set_title("2005 GDP Per Person vs. Human Capital")
ax.set_ylabel("Log Human Capital") 
ax.set_xlabel("GDP Per Person, PPP 2005 Dollars")

xlabel_list = np.exp(range(5,13)) # Now creat the list of lables by converting 5,6,etc. to levels
                                  # by taking exp.
xlabel_list = np.round(xlabel_list,-2) # Then round it so it looks nice.


ax.set_xticklabels(xlabel_list) # Then set the xtick labels.

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.savefig("hc.png", bbox_inches="tight", dpi = 600)

plt.show()

# Here is an example to get a handle on this:

data_ex = {"Year": [2010,2010,2011,2011],
            "GDP": [1 , 2 , 3, 4],
            "INV": [10, 12, 13, 14],
            "CNT": ["USA", "ARG", "USA", "ARG"]}

weo  = pd.DataFrame(data_ex)

print(weo.groupby("Year").sum())
print(weo.groupby("Year").mean())

# This saying take the data frame, form a group by individual
# years, then sum accross the group. 

avg = pwt.groupby("year").gdp_pop.median() # This will compute the median, across countries, within a year

fig, ax  = plt.subplots()

avg.plot(ax = ax)

plt.show()

med = np.log(pwt.groupby("year").gdp_pop.median()) # This will compute the median, across countries, within a year
q90 = np.log(pwt.groupby("year").gdp_pop.quantile(0.90)) # This is going to compute the 90th percentile
q10 = np.log(pwt.groupby("year").gdp_pop.quantile(0.10)) # This is going to compute teh 10th percentile

# I'm doing this in logs for lots of reasons, try it without...

fig, ax  = plt.subplots()

med.plot(ax = ax)

q90.plot(ax = ax)

q10.plot(ax = ax)

plt.show()

#ax.fill_between(med.index, q10, q90, color = "b")

fig, ax  = plt.subplots()

med.plot(ax = ax, color = "white", lw = 3, figsize = (7,5))

ax.fill_between(med.index, q10, q90, color = "#3F5D7D") 
# This is the new option, we specify the x-axis, in this case the index,
# color in the areas between the bottom q10
# and the top, q90. Then I got this color from Randy olson blog, I like it.

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

###############################################################
# This is the new part

ax.plot(med.index, np.log(pwt[pwt.countrycode == "KOR"].gdp_pop), color = 'r', lw = 3)

# just plot on top, the x-axis, Korea's GDP per person, color, line width...

###############################################################

ax.set_yticks(range(5,12))
ax.set_xlim(1950,2012)
# This sets the yticks so we can see center the graph in the middle

y_label_list = np.exp(range(5,12))
y_label_list = np.round(y_label_list,-2)

ax.set_yticklabels(y_label_list)
# This process above generates informative labels for the y-axis
# so its not in log units, but in dollars

ax.set_title("Global Income Inequality\n") 
ax.set_ylabel("GDP Per Person, PPP 2005 Dollars")
ax.set_xlabel("Year")

ax.text(1960, 3.75, "Data Source: Penn World Table 8.1, " 
        "http://www.rug.nl/ggdc/docs/pwt81.xlsx", fontsize = 8)

plt.show()

plt.style.use('dark_background')
#plt.xkcd()
# Here are some styles to trr
# 'fivethirtyeight' ,ggplot, bmh, dark_background, and grayscale

fig, ax  = plt.subplots()

med.plot(ax = ax, color = "white", lw = 3, figsize = (7,5))

ax.fill_between(med.index, q10, q90, color = "#3F5D7D", alpha = 0.5) 
# This is the new option, we specify the x-axis, in this case the index,
# color in the areas between the bottom q10
# and the top, q90. Then I got this color from Randy olson blog, I like it.

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

###############################################################
# This is the new part

ax.plot(med.index, np.log(pwt[pwt.countrycode == "KOR"].gdp_pop), color = 'r', lw = 3)

# just plot on top, the x-axis, Korea's GDP per person, color, line width...

###############################################################

ax.set_yticks(range(5,12))
ax.set_xlim(1950,2012)
# This sets the yticks so we can see center the graph in the middle

y_label_list = np.exp(range(5,12))
y_label_list = np.round(y_label_list,-2)

ax.set_yticklabels(y_label_list)
# This process above generates informative labels for the y-axis
# so its not in log units, but in dollars

ax.set_title("Global Income Inequality\n") 
ax.set_ylabel("GDP Per Person, PPP 2005 Dollars")
ax.set_xlabel("Year")

ax.text(1960, 3.75, "Data Source: Penn World Table 8.1, " 
        "http://www.rug.nl/ggdc/docs/pwt81.xlsx", fontsize = 8)

plt.show()

mpl.rcParams.update(mpl.rcParamsDefault)

get_ipython().magic('matplotlib inline')



