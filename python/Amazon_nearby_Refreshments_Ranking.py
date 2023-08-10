# Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

restaurant = pd.read_csv('../Results/Restaurant_Rating.csv')
del restaurant['Unnamed: 0']
del restaurant['restaurant Total Count']
del restaurant['price_level']
restaurant['Rating']=restaurant['Rating'].astype(float)
restaurant.replace('NAN', value=0, inplace=True)
restaurant = restaurant.rename(columns={'Facility restaurant':'EatingOut Facility'})
restaurant['Type']='Restaurant'
restaurant.head()

####TEST
#restaurant.groupby('City Name').median()

cafe = pd.read_csv('../Results/Cafe_Rating.csv')
del cafe['Unnamed: 0']
del cafe['cafe Total Count']
del cafe['price_level']
cafe['Rating']=cafe['Rating'].astype(float)
cafe.replace('NAN', value=0, inplace=True)
cafe = cafe.rename(columns={'Facility cafe':'EatingOut Facility'})
cafe['Type']='Cafe'
cafe.head()

new_eatingout_df = restaurant.append(cafe)
new_eatingout_df.reset_index(drop=True)
new_eatingout_df.head()

print("===================TEST===================")
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)

# Set figure width to 12 and height to 9
fig_size[0] = 30
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size


ranks = new_eatingout_df.groupby("City Name")["Rating"].median().fillna(0).sort_values()[::-1].index
print(ranks)
ranks_dict = {}
y = 1
for x in ranks:
    ranks_dict[x] = y
    y=y+1
    
print(ranks_dict)

ax = sns.boxplot(x='City Name', y='Rating', data=new_eatingout_df, width=.5, order=ranks)

# Set scale for all the fonts of the plot
sns.set(font_scale=1.8)

plt.ylim(1,5.5)
# Make x-axis, y-axis & title labels
ax.set_title("City Refreshments Ranking", fontsize=35, fontweight='bold')
ax.set_xlabel(" ", fontsize=20)
ax.set_ylabel("Rating", fontsize=30, verticalalignment='bottom', horizontalalignment='center')

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

nobs = [str(x) for x in ranks_dict.values()]
print(nobs)
nobs = ["#" + i for i in nobs]
print(nobs)
# Add it to the plot
pos = range(len(nobs))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], 5.2, nobs[tick],
            horizontalalignment='center', size='large', color='black', weight='semibold')


print("===========================================")
print("====================END====================")
print(' ')
print(' ')
plt.savefig('Save_Figs/Refreshments.png')
plt.show()



