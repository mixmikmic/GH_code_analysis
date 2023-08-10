import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/mwaugh0328/Data_Bootcamp_Fall_2017/master/data_bootcamp_exam/baseball_exam_data.csv"
nl = pd.read_csv(url)

nl.shape

nl.columns

nl.dtypes
# Note how therea are two types. One is an integer
# and another is an object. The object basically picks up
# anythhing that can not be classified as a numerical (float or int) type

westdivision = nl.name.tail(5)

# These are the names of the last five baseball teams in the data set
# they happen to correspons with the nl west division...

nl.set_index("name", inplace = True)

nl["BatAverage"] = nl.Hits / nl.AtBats
nl["HRAverage"] = nl.HR / nl.AtBats

print("Mean Batting Average", nl.BatAverage.mean())
print("Median Batting Average", nl.HRAverage.median())

corr_mat = nl.corr()

print("Correlation of Wins and Attendence", corr_mat.Wins.attendance)

nl.BatAverage.sort_values().plot(kind = "barh", xlim = (0.22,0.28), color = "blue", alpha = 0.75)
plt.show()

fig, ax = plt.subplots(1,2, figsize = (7,5))

fig.tight_layout(w_pad=5) 

ax[0].scatter(nl.Wins, nl.BatAverage, s = 0.00025*nl.attendance, alpha = 0.5)
ax[1].scatter(nl.Wins, nl.HRAverage, s = 0.00025*nl.attendance, alpha = 0.5)

ax[0].set_ylim(0.22,0.28)
ax[1].set_ylim(0.01,0.04)

ax[0].set_xlabel("Wins")
ax[1].set_xlabel("Wins")

ax[0].set_ylabel("Batting Average")
ax[1].set_ylabel("HR Average")

plt.show()





