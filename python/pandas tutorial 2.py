import pandas as pd
import numpy

#This playground is inspired by Greg Reda's post on Intro to Pandas Data Structures:
#http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/

if True:
    data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
            'wins': [11, 8, 10, 15, 11, 6, 10, 4],
            'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
    football = pd.DataFrame(data)
    print "we will print only the years: "
    print football['year']
    print ''
    print "only the years too, bu using the shortcut: dataframename.keyname"
    print football.year  # shorthand for football['year']
    print ''
    print "from the data frame we print years, wins and losses:"
    print football[['year', 'wins', 'losses']]

#Row selection can be done through multiple ways.

#Some of the basic and common methods are:
#  1) Slicing
# 2) An individual index (through the functions iloc or loc)
# 3) Boolean indexing

#You can also combine multiple selection requirements through boolean
#operators like & (and) or | (or)

# Change False to True to see boolean indexing in action
if True:
    data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
            'wins': [11, 8, 10, 15, 11, 6, 10, 4],
            'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
    football = pd.DataFrame(data)
    print "here the first column of the dataframe (iloc)"
    print football.iloc[[0]]
    print ""
    print "here the first column of the dataframe, using a different operator (loc)"
    print football.loc[[0]]
    print ""
    print "here please return the columns from position 3 to 5(exclusive), so we have the column index 3, 4"
    print football[3:5]
    print ""
    print "print only the rows where column was bigger than 10"
    print football[football.wins > 10]
    print ""
    print "print only the rows where column is bigger than 10 and the team is packers"
    print football[(football.wins > 10) & (football.team == "Packers")]

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]
    
olympic_medal_counts = {'country_name': pd.Series(countries),
                            'gold': pd.Series(gold),
                            'silver': pd.Series(silver),
                            'bronze': pd.Series(bronze)}
df = pd.DataFrame(olympic_medal_counts)
    
# YOUR CODE HERE
#goldOne = df[(df.gold >= 1)]
#avg_bronze_at_least_one_gold = numpy.mean(goldOne['bronze'])

#other way...
bronze_at_least_one_gold = df['bronze'] [df['gold'] >= 1]
avg_bronze_at_least_one_gold = numpy.mean(bronze_at_least_one_gold)

avg_bronze_at_least_one_gold

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
             'Netherlands', 'Germany', 'Switzerland', 'Belarus',
             'Austria', 'France', 'Poland', 'China', 'Korea', 
             'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
             'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
             'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

olympic_medal_counts = {'country_name':countries,
                        'gold': pd.Series(gold),
                        'silver': pd.Series(silver),
                        'bronze': pd.Series(bronze)}    
df = pd.DataFrame(olympic_medal_counts)

# YOUR CODE HERE
#meanS = pd.Series([numpy.mean(df.gold), numpy.mean(df.silver), numpy.mean(df.bronze)])
#ndf = df[['gold','silver','bronze']]
#ndfAve = ndf.apply(numpy.mean)
#ndfAve

#other way is:
ndf = df[['gold','silver','bronze']].apply(numpy.mean)
ndf

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

#your code here
#df = {'country_name': countries, 
#      'points': pd.Series(numpy.dot(gold, 4) + numpy.dot(silver, 2) + pd.Series(bronze))}

#olympic_points_df = pd.DataFrame(df)
#olympic_points_df

#other way.
#first a data frame
data = {'countries': pd.Series(countries), 'gold': pd.Series(gold), 'silver': pd.Series(silver), 'bronze':pd.Series(bronze)}
df = pd.DataFrame(data)

medals = df[['gold', 'silver', 'bronze']]
mat = numpy.dot(medals, [4, 2, 1])

ndf = pd.DataFrame({'countries': countries, 'points': mat})
ndf



