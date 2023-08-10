get_ipython().run_line_magic('pylab', 'inline')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

categories_vs_reviews = pd.read_csv('business-categories-reviews.csv')
top_50_categories_vs_reviews = categories_vs_reviews.pivot_table('Review Count', index='Category').sort_values(by='Review Count', ascending=False)[:30]
print("Business Categories wit maximum Reviews")
display(top_50_categories_vs_reviews.head(10))
print (len(categories_vs_reviews))

graph = top_50_categories_vs_reviews.plot(kind='bar',width = 0.35,figsize=(16,8))
graph.set_ylabel('Review Count',fontsize=18)
graph.set_xlabel('Category',fontsize=18)
graph.set_title('Title',fontsize=18)
for tick in graph.get_xticklabels():
    tick.set_fontsize("20")
for tick in graph.get_yticklabels():
    tick.set_fontsize("20")
plt.show()

# libraries and data
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pprint
from tabulate import tabulate

def plot_line(df,header):
    # style
    plt.style.use('seaborn-darkgrid')

    # line plot
    # first is x axis, 2nd is y axis

    plt.plot(df[str(header[0])], df[str(header[1])], marker='', color='red', linewidth=1, alpha=1)

    # Add legend

    red_line = mlines.Line2D([], [], color='red', alpha=1, linewidth=2, label=str(header[1]))
    plt.legend(loc=1, ncol=2, handles=[red_line])
    #red_patch = mpatches.Patch(color='red', label=header[1])
    #plt.legend(loc=1, ncol=2, handles=[red_patch])


    # Add titles
#     plt.title("Frequency", loc='left', fontsize=14, fontweight=0, color='orange')
    plt.xlabel(str(header[0]))
    plt.ylabel(str(header[1]))

    #plt.xticks(df[str(header[0])] , rotation=45 )
    plt.show(block=True)

df = pd.DataFrame(
    {'Business Categories': list(range(1,len(categories_vs_reviews)+1)),
     'No. of Reviews': categories_vs_reviews['Review Count']
    })
display(df.head())

plot_line(df,list(df))

state_vs_reviews = pd.read_csv('states-reviews.csv')
top_50_state_vs_reviews = (state_vs_reviews.pivot_table('Review Count', index='State')
                           .sort_values(by='Review Count', ascending=False)[:30])
print(" Top 10 states with max reviews")
display(top_50_state_vs_reviews.head(10))
print (len(state_vs_reviews))

graph = top_50_state_vs_reviews.plot(kind='bar',width = 0.35,figsize=(16,8))
graph.set_ylabel('Review Count',fontsize=18)
graph.set_xlabel('State',fontsize=18)
# graph.set_title('Title',fontsize=18)
for tick in graph.get_xticklabels():
    tick.set_fontsize("18")
for tick in graph.get_yticklabels():
    tick.set_fontsize("18")
plt.show()

def plot_line2(X,Y,x_label,y_label,title,legend):
    # style
    plt.style.use('seaborn-darkgrid')

    # line plot
    # first is x axis, 2nd is y axis

    plt.plot(X, Y, marker='', color='red', linewidth=1, alpha=1)

    # Add legend

    red_line = mlines.Line2D([], [], color='red', alpha=1, linewidth=2, label=legend)
    plt.legend(loc=1, ncol=2, handles=[red_line])
    #red_patch = mpatches.Patch(color='red', label=header[1])
    #plt.legend(loc=1, ncol=2, handles=[red_patch])


    # Add titles
    plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    #plt.xticks(df[str(header[0])] , rotation=45 )
    plt.show(block=True)

df2 = pd.DataFrame(
    {
     'No. of Reviews': state_vs_reviews['Review Count'],
     'State': list(range(1,len(state_vs_reviews)+1))
    })

title = "Frequency of no. of reviews according to states"
header = list(df2)
plot_line2(df2[str(header[1])],df2[str(header[0])],str(header[1]),str(header[0]),title,str(header[0]))

cities_vs_reviews = pd.read_csv('cities-reviews.csv')
top_50_cities_vs_reviews = cities_vs_reviews.pivot_table('Review Count', index='City').sort_values(by='Review Count', ascending=False)[:30]
display(top_50_cities_vs_reviews.head(10))

graph = top_50_cities_vs_reviews.plot(kind='bar',width = 0.35,figsize=(16,8))
graph.set_ylabel('Review Count',fontsize=18)
graph.set_xlabel('City',fontsize=18)
# graph.set_title('Title',fontsize=18)
for tick in graph.get_xticklabels():
    tick.set_fontsize("18")
for tick in graph.get_yticklabels():
    tick.set_fontsize("18")
plt.show()

df3 = pd.DataFrame(
    {'City': list(range(1,len(cities_vs_reviews)+1)),
     'No. of Reviews': cities_vs_reviews['Review Count']
    })

plot_line(df3,list(df3))

