import agate

matches = agate.Table.from_csv('english_county_championship_innings.csv')

print matches

grounds = matches.where(lambda row: row['month'] == 4 or row['month'] == 5).where(lambda row: row['year'] > 2010).group_by('ground')

ground_avg = grounds.aggregate([('first_innings_avg', agate.Mean('first_innings_runs'))])

ground_avg = ground_avg.order_by('first_innings_avg', reverse=True)

ground_avg.print_table(max_column_width=50)

prior_years = matches.where(lambda row: row['month'] == 4 or row['month'] == 5).where(lambda row: row['year'] < 2011 and row['year'] > 2005).group_by('ground')

prior_avg = prior_years.aggregate([('first_innings_avg', agate.Mean('first_innings_runs')), ('matches', agate.Count())])

prior_avg = prior_avg.order_by('first_innings_avg', reverse=True)

prior_avg = prior_avg.where(lambda row: row['matches'] > 5)

prior_avg.print_table(max_column_width=50)



