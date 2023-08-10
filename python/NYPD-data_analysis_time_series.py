import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser
import datetime
get_ipython().magic('matplotlib inline')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def read_in_initial(year, filename):
    df = pd.read_csv(filename, encoding = "ISO-8859-1", low_memory=False)
    df_race = pd.DataFrame(df['race'].value_counts())
    return df_race

df_2015 = read_in_initial(2015, '2015.csv')
df_2015.columns = ['2015']
df_2015['race']=df_2015.index
df_2015

years_list = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

totals_list = []
for year in years_list:
#Read in the data for each year
    df_year = pd.read_csv(year+'.csv', encoding = "ISO-8859-1", low_memory=False)
#get the total number of stop records for a year
    total = len(df_year['year'])
#create a dictionary for each year, containing the year and total number of stops
    dict_t = {'Year': year, 'total_count': total}
#append that dict to a list collecting the dicts for each year
    totals_list.append(dict_t)
#make a new dataframe that holds the value counts for each race
    df_race = pd.DataFrame(df_year['race'].value_counts())
#name the column with that counts after the year they are from
    df_race.columns = [year]
#make the 'race' column the index...
    df_race['race']=df_race.index
#...so you can merge on it with a df previously defined outside of this function
    df_2015 = df_2015.merge(df_race, left_on='race', right_on='race')

new_df_autom = df_2015.copy()
new_df_autom

totals_list

totals_df = pd.DataFrame(totals_list)
totals_df

def convert_year(year_int):
    year_str = str(year_int)
    parsed_year = dateutil.parser.parse(year_str+'-01-01')
    return parsed_year

new_totals = totals_df.copy()
parsed_year_index = [convert_year(year) for year in years_list]
new_totals['parsed'] = parsed_year_index

new_totals

new_df_autom.index = new_df_autom['race']
new_df_autom = new_df_autom.drop('race', 1)
new_df_autom = new_df_autom.drop('2015_x', 1)

new_df_autom.columns = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',
       '2012', '2013', '2014', '2015']

new_df_autom

df_trans = new_df_autom.transpose()

df_trans

number_list = df_trans.index 
df_new = df_trans.copy()
parsed_index = [convert_year(number) for number in number_list]
df_new['parsed'] = parsed_index

df_new.set_index(df_new['parsed'])

merged_total = df_new.merge(new_totals, left_on='parsed', right_on='parsed')

merged_total

merged_total.index = merged_total['parsed']

merged_total.plot(x='parsed', y= 'total_count')

# constructed_df = pd.read_csv('constructed_dataset.csv')

constructed_df = merged_total.copy()
constructed_df = constructed_df.drop('parsed', 1)
constructed_df = constructed_df.drop('I', 1)
constructed_df = constructed_df.drop('Z', 1)
constructed_df = constructed_df.drop('A', 1)
constructed_df = constructed_df.drop('P', 1)
constructed_df['b_q_w'] = constructed_df['B'] + constructed_df['Q'] + constructed_df['W']
constructed_df['others'] = constructed_df['total_count'] - constructed_df['b_q_w']
constructed_df = constructed_df.drop('b_q_w', 1)
constructed_df

constructed_df.columns = ('Black', 'Hispanic', 'White', 'Year', 'Total Stops', 'All_others')

constructed_df

constructed_df.plot()

fig, ax = plt.subplots(figsize=(12,5))
ax = constructed_df['Total Stops'].plot(c='grey', linewidth=3)

ax.set_xticks(['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012', '2013', '2014','2015'])
ax.set_xticklabels(['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012', '2013', '2014','2015'])
ax.set_xlim((pd.Timestamp('2003-01-01'), pd.Timestamp("2015-01-01")))

#get rid of the frame around it
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

#tackling ticks
plt.tick_params(
which='major',
top='off',
left='off',
right= 'off',
bottom='off',
labeltop='off',
labelbottom='on')

#When plotting a grid, you can pass it options
ax.grid(True)

#When plotting a grid, you can pass it options
ax.grid(color='lightgrey', linestyle=':', linewidth=1)

#send grid to the back
ax.set_axisbelow(True)

ax.set_title("Sharp decline in NYPD stop-and-frisk")
ax.set_xlabel("Year")
ax.set_ylabel("Number of stops per year")

plt.savefig("NYPD_stop_frisk_timeseries.pdf", transparend=True)

## CODE COPIED FROM CHRIS ALBON: http://chrisalbon.com/python/matplotlib_percentage_stacked_bar_plot.html

# Create a figure with a single subplot
f, ax = plt.subplots(1, figsize=(10,5))

# Set bar width at 1
bar_width = 1

# positions of the left bar-boundaries
bar_l = [i for i in range(len(constructed_df['Black']))] 

tick_pos = [i+(bar_width/2) for i in bar_l] 

totals = [i+j+k+l for i,j,k,l in zip(constructed_df['Black'], constructed_df['Hispanic'], constructed_df['White'], constructed_df['All_others'])]

black_rel = [i / j * 100 for  i,j in zip(constructed_df['Black'], totals)]

hisp_rel = [i / j * 100 for  i,j in zip(constructed_df['Hispanic'], totals)]

white_rel = [i / j * 100 for  i,j in zip(constructed_df['White'], totals)]

others_rel = [i / j * 100 for  i,j in zip(constructed_df['All_others'], totals)]

ax.bar(bar_l, 
       # using black_rel data
       black_rel, 
       # labeled 
       label='Black', 
       # with alpha
       alpha=0.6, 
       # with color
       color='#404e7c',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l, 
       # using mid_rel data
       hisp_rel, 
       # with pre_rel
       bottom=black_rel, 
       # labeled 
       label='Hispanic', 
       # with alpha
       alpha=0.6, 
       # with color
       color='#fe5f55', 
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l, 
       # using post_rel data
       white_rel, 
       # with pre_rel and mid_rel on bottom
       bottom=[i+j for i,j in zip(black_rel, hisp_rel)], 
       # labeled 
       label='White',
       # with alpha
       alpha=0.6, 
       # with color
       color='#e8d245', 
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l, 
       # using post_rel data
       others_rel, 
       # with pre_rel and mid_rel on bottom
       bottom=[i+j+k for i,j,k in zip(black_rel, hisp_rel, white_rel)], 
       # labeled 
       label='Others',
       # with alpha
       alpha=0.6, 
       # with color
       color='#c6caed', 
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Set the ticks to be first names
plt.xticks(tick_pos, constructed_df['Year'])
ax.set_ylabel("Racial breakdown of total people stopped [in percent]")
ax.set_xlabel("Year")

ax.set_title("NYPD stop-and-frisk: Racial bias remains")

# Let the borders of the graphic
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim(-10, 110)

# rotate axis labels
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

# shot plot
plt.show()

f.savefig("NYPD_racial_bias_CORRECTED.pdf", transparent=True)



