import pandas as pd
import csv
import warnings
# warnings.simplefilter(action = "ignore", category = SettingWithCopyWarning)
warnings.simplefilter("ignore", pd.core.common.SettingWithCopyWarning)

raw_business = pd.read_csv('./dataset/yelp_academic_dataset_business.csv', sep=',', index_col=None )
print 'shape=',raw_business.shape
print raw_business.columns
raw_business.head()

raw_review = pd.read_csv('./dataset/yelp_academic_dataset_review.csv', sep=',', index_col=None )
print 'shape=',raw_review.shape
print raw_review.columns
raw_review.head()

review_of_business = raw_review[['business_id','text','stars','date','votes.cool','votes.funny','votes.useful']]
print 'shape=',review_of_business.shape
print review_of_business.columns
review_of_business.head()

raw_tip = pd.read_csv('./dataset/yelp_academic_dataset_tip.csv', sep=',', index_col=None )
print 'shape=',raw_tip.shape
print raw_tip.columns
raw_tip.head()



raw_checkin = pd.read_csv('./dataset/yelp_academic_dataset_checkin.csv', sep=',', index_col=None )
print 'shape=',raw_checkin.shape
print raw_checkin.columns
raw_checkin.head()

raw_user = pd.read_csv('./dataset/yelp_academic_dataset_user.csv', sep=',', index_col=None )
print 'shape=',raw_user.shape
print raw_user.columns
raw_user.head()

#bar plot
location_star = raw_business[['state','stars']]
star_for_state = location_star.groupby(['state'])
state_avg_star = star_for_state.mean()
state_avg_star['count'] = star_for_state.count()
print state_avg_star.shape, '= (x,y), x = Number of states ',
state_avg_star 

state_avg_star = state_avg_star[ state_avg_star['count']>10 ] # exclude rows that only has 3 count
state_avg_star = state_avg_star.sort_values(['stars'], ascending=False)
print state_avg_star.shape
state_avg_star

### Setup for data visualization
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

### You need to create a plotly account.
### Use the username API key  
plotly.tools.set_credentials_file(username='user', api_key='your key')

data = [go.Bar(
            x=state_avg_star.index,
            y=state_avg_star['stars']
    )]
layout = go.Layout(
    title='Average Stars for Each State',
    xaxis=dict(
        title='State names',
        tickangle=45,
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        title='# of Stars',
        titlefont=dict(color='#1f77b4'),
        tickfont=dict(size=10,color='#1f77b4'), 
#         tick0 = 2,
#         dtick = 0.2
    ),    
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='AVG_star_for_state')

location_count = raw_business[['state','city','stars']]
location_count = location_count.groupby(['state','city'])
location_stat = location_count.count()
location_stat = location_stat[ location_stat['stars']>10 ]
print location_stat.shape, '= (x,y), x = Number of cities '
location_stat

location_count = raw_business[['state','stars']]
location_count

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))   # set the size of figure (x,y)

# durationUFO = cleanUFO.Duration.astype(float) 
# shapeUFO = cleanUFO.Shape 
ax = sns.boxplot(x=location_count['stars'], groupby=location_count['state']) 
# ax.set_yscale('log')
plt.xlabel('States', fontsize=20)
plt.ylabel('# of Star', fontsize=20)
plt.title('The Distribution of Stars for each State', fontsize=24)
plt.show()



category_stars = raw_business[['categories','stars']]
category_stars.head()

US_tradition = category_stars[ category_stars['categories'].str.contains('Traditional')]
US_new = category_stars[ category_stars['categories'].str.contains('New')]
Mexican = category_stars[ category_stars['categories'].str.contains(u'Mexican')]
Indian = category_stars[ category_stars['categories'].str.contains(u'Indian')]
Italian = category_stars[ category_stars['categories'].str.contains(u'Italian')]
Chinese = category_stars[ category_stars['categories'].str.contains(u'Chinese')]

Cuban = category_stars[ category_stars['categories'].str.contains(u'Cuban')]
Brunch = category_stars[ category_stars['categories'].str.contains(u'Breakfast & Brunch')]
Fish = category_stars[ category_stars['categories'].str.contains(u'Fish & Chips')]
Mediterranean = category_stars[ category_stars['categories'].str.contains(u'Mediterranean')]

Mediterranean
# burgers.describe()


x_data = ['American (Traditional)', 'American (New)','Mexican', 'Indian','Italian','Chinese',
         'Cuban', 'Brunch', 'Fish & Chips', 'Mediterranean']

y_data = [US_tradition.stars, US_new.stars, Mexican.stars, Indian.stars, Italian.stars, Chinese.stars, 
          Cuban.stars, Brunch.stars, Fish.stars, Mediterranean.stars]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(123, 34, 23, 0.5)', 'rgba(27,66, 45, 0.5)', 'rgba(73, 126, 12, 0.5)', 'rgba(67, 210, 56, 0.5)']

traces = []

for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
#             boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Stars Distribution by Types of Restaurants',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig,filename='Stars Distribution by Types of Restaurants')



print "Total number of business in the dataset: ", category_stars.shape
print 'Total number of unique categories: ', len( category_stars.categories.unique() )

category_find = raw_business[['categories','stars']]

category_count = []
category_name = ['Active Life', 'Arts & Entertainment','Automotive','Beauty & Spas','Education',
                'Event Planning & Services','Financial Services','Food','Health & Medical','Home Services',
                'Hotels & Travel','Local Flavor','Local Services','Mass Media','Nightlife',
                'Pets','Professional Services','Public Services & Government','Real Estate','Religious Organizations',
                'Restaurants','Shopping']

category_count.append( category_find[ category_find['categories'].str.contains('Active Life')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Arts & Entertainment')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Automotive')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Beauty & Spas')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Education')].count() )

category_count.append( category_find[ category_find['categories'].str.contains('Event Planning & Services')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Financial Services')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Food')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Health & Medical')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Home Services')].count() )

category_count.append( category_find[ category_find['categories'].str.contains('Hotels & Travel')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Local Flavor')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Local Services')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Mass Media')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Nightlife')].count() )

category_count.append( category_find[ category_find['categories'].str.contains('Pets')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Professional Services')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Public Services & Government')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Real Estate')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Religious Organizations')].count() )

category_count.append( category_find[ category_find['categories'].str.contains('Restaurants')].count() )
category_count.append( category_find[ category_find['categories'].str.contains('Shopping')].count() )

category_stat = pd.DataFrame.from_dict(category_count)
category_stat = category_stat.rename(columns = {'categories':'Counts'})

category_stat.index = category_name
category_stat

# Figure: How many categories are in Yelp reviews
fig_category = {
    'data': [{'labels': category_stat.index,
              'values': category_stat.Counts,
              'type': 'pie'}],
    'layout': {'title': 'The categories of business that are reviewed in Yelp'}
     }

py.iplot(fig_category, filename='counts_of_categories')

restaurants = category_find[ category_find['categories'].str.contains('Restaurants')]
print 'Total number of unique categories: ', len( restaurants.categories.unique() )

# number of stars of all business
stars_all = raw_business[['stars','name']]
star_distribute = stars_all.groupby(['stars']).count()
print "Mean value of the stars = ",raw_business['stars'].mean() 
star_distribute = star_distribute.rename(columns = {'name':'Counts'})
# star_distribute = star_distribute.sort_values(['stars'], ascending=False)
star_distribute

# Figure:Star distribution in Yelp business
fig_star_distribute = {
    'data': [{'labels': star_distribute.index,
              'values': star_distribute.Counts,
              'type': 'pie'}],
    'layout': {'title': 'Stars Distribution of all the reviewed business in Yelp'}
     }

py.iplot(fig_star_distribute, filename='Satrs_distribution')



trace_backers = go.Bar(
    x= star_distribute.index,
    y= star_distribute.Counts,
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(color='rgba(55, 128, 191, 1.0)',width=2)),
        
)

data_backers = [trace_backers]
layout = go.Layout(
    title='Star vs. Number of Business',
    font=dict(family='Arial', size=18,color='rgb(0, 0, 0)'),
    xaxis=dict(
        title='Star Rating',
        titlefont=dict(
            size=16,
            color='rgb(15, 15, 15)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Number of Bsuiness',
        titlefont=dict(
            size=16,
            color='rgb(15, 15, 15)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    bargap=0.15,
    annotations = [dict(
            x=xi,
            y=yi,
            text=str(format(yi,',')),
            font=dict(family='Arial', size=14,color='rgba(66, 119, 244, 1)'),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
        ) for xi, yi in zip(star_distribute.index, star_distribute.Counts)]
)

fig_backers = go.Figure(data=data_backers, layout=layout)
py.iplot(fig_backers, filename='basic-bar')

# number of stars in top 10 categories
stars_category01 = category_find[ category_find['categories'].str.contains('Active Life')].stars
stars_category02 = category_find[ category_find['categories'].str.contains('Arts & Entertainment')].stars
stars_category03 = category_find[ category_find['categories'].str.contains('Automotive')].stars
stars_category04 = category_find[ category_find['categories'].str.contains('Beauty & Spas')].stars
stars_category05 = category_find[ category_find['categories'].str.contains('Education')].stars

stars_category06 = category_find[ category_find['categories'].str.contains('Event Planning & Services')].stars
stars_category07 = category_find[ category_find['categories'].str.contains('Financial Services')].stars
stars_category08 = category_find[ category_find['categories'].str.contains('Food')].stars
stars_category09 = category_find[ category_find['categories'].str.contains('Health & Medical')].stars
stars_category10 = category_find[ category_find['categories'].str.contains('Home Services')].stars

stars_category11 = category_find[ category_find['categories'].str.contains('Hotels & Travel')].stars
stars_category12 = category_find[ category_find['categories'].str.contains('Local Flavor')].stars
stars_category13 = category_find[ category_find['categories'].str.contains('Local Services')].stars
stars_category14 = category_find[ category_find['categories'].str.contains('Mass Media')].stars
stars_category15 = category_find[ category_find['categories'].str.contains('Nightlife')].stars

stars_category16 = category_find[ category_find['categories'].str.contains('Pets')].stars
stars_category17 = category_find[ category_find['categories'].str.contains('Professional Services')].stars
stars_category18 = category_find[ category_find['categories'].str.contains('Public Services & Government')].stars
stars_category19 = category_find[ category_find['categories'].str.contains('Real Estate')].stars
stars_category20 = category_find[ category_find['categories'].str.contains('Religious Organizations')].stars

stars_category21 = category_find[ category_find['categories'].str.contains('Restaurants')].stars
stars_category22 = category_find[ category_find['categories'].str.contains('Shopping')].stars

# category_name = ['Active Life', 'Arts & Entertainment','Automotive','Beauty & Spas','Education',
#                 'Event Planning & Services','Financial Services','Food','Health & Medical','Home Services',
#                 'Hotels & Travel','Local Flavor','Local Services','Mass Media','Nightlife',
#                 'Pets','Professional Services','Public Services & Government','Real Estate','Religious Organizations',
#                 'Restaurants','Shopping']
# star_list =[]
# for i in range( 0, len(category_name) ): 
#      star_list[i] = category_find[ category_find['categories'].str.contains( category_name[i] )].stars


star_x = category_name

star_y = [stars_category01, stars_category02, stars_category03, stars_category04, stars_category05,  
          stars_category06, stars_category07, stars_category08, stars_category09, stars_category10, 
          stars_category11, stars_category12, stars_category13, stars_category14, stars_category15, 
          stars_category16, stars_category17, stars_category18, stars_category19, stars_category20, 
          stars_category21, stars_category22]

# colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
#          'rgba(123, 34, 23, 0.5)', 'rgba(27,66, 45, 0.5)', 'rgba(73, 126, 12, 0.5)', 'rgba(67, 210, 56, 0.5)']

traces = []

for xd, yd in zip(star_x, star_y):
        traces.append(go.Box(
            y=yd,
            name=xd,
#             boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
#             fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Stars Distribution for each category',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=120,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig,filename='Stars Distribution for each category')


# find business_id by indicating the name of category
# Enter the name of categories that you would like to find

BID_by_category = raw_business[['categories','business_id','stars']]
target = u'Breakfast & Brunch' # Enter the name of categories that you would like to find
BID_by_category[ BID_by_category['categories'].str.contains(target)]







