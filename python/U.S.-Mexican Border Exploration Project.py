from datascience import * # a UC Berkeley developed wrapper of Pandas library for beginning students in Data Science
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

#from client.api.notebook import Notebook
#ok = Notebook('exploration.ok')
#_ = ok.auth(inline=True)

apprehensions = Table().read_table('Apprehensions.csv')
apprehensions

wall = Table().read_table('the-wall.csv')
wall.sort('year')

staffing = Table().read_table('Staffing.csv')
staffing = staffing.with_column("Year", np.arange(1992, 2017))
staffing.show()

walls_with_years = wall.where('year', are.not_equal_to('TBD'))
cleaned_wall = walls_with_years.with_column('year_0', walls_with_years.column('year').astype(np.int)).drop('year')
cleaned_wall_with_miles = cleaned_wall.with_column('miles', cleaned_wall.column('length_ft') / 5280)
cleaned_wall_with_miles

cleaned_wall_with_miles.group('year_0', sum).plot('year_0', 'miles sum')

wall_type_bar = cleaned_wall_with_miles.group('vbr_type', sum).barh('vbr_type','miles sum')
wall_type_bar

wall_type_bar = cleaned_wall_with_miles.group('wall_type', sum).barh('wall_type','miles sum')
wall_type_bar

staffing

apprehensions_and_wall = cleaned_wall_with_miles.group('year_0', sum).join(
    'year_0', apprehensions, 'Fiscal Year').select([0, 1, 2, 3, 4, 14]).relabeled(
    'Southwest Border Total', 'Apprehensions Total')

apprehensions_and_wall_and_staffing = apprehensions_and_wall.join(
    'year_0', staffing, 'Year').select([0, 3, 4, 5, 15]).relabeled('Total', 'Staffing')

apprehensions_and_wall_and_staffing.show()

apprehensions_and_wall_and_staffing.plot("year_0", "Apprehensions Total")

apprehensions_and_wall_and_staffing.plot('year_0', 'Staffing')

apprehensions_and_wall_and_staffing.with_column('Apprehensions/Staffing', apprehensions_and_wall_and_staffing.column('Apprehensions Total')/apprehensions_and_wall_and_staffing.column('Staffing')).plot('year_0', 'Apprehensions/Staffing')

apprehensions_and_wall_and_staffing.with_column(
    "Staffing x 100",
    apprehensions_and_wall_and_staffing.column(4) * 100).drop(1,2,4).plot("year_0")

apprehensions_and_wall_and_staffing.select(0,1,3).move_to_start(
    "Apprehensions Total").plot("year_0")

apprehensions_and_wall_and_staffing.with_column(
    "Staffing x 100",
    apprehensions_and_wall_and_staffing.column(4) * 100).select(
    0,1,3,5).move_to_start("Apprehensions Total").plot("year_0")

just_app_and_wall = apprehensions_and_wall_and_staffing.select("miles sum", "Apprehensions Total")

just_app_and_wall.scatter("miles sum", "Apprehensions Total")

just_app_and_staffing = apprehensions_and_wall_and_staffing.select('Staffing', 'Apprehensions Total')
just_app_and_staffing.scatter(0, 1)

def standard_units(arr):
    standard_units = (arr - np.mean(arr)) / np.std(arr)
    return standard_units 

def correlation(tbl):
    first_column_su = standard_units(tbl.column(0))
    second_column_su = standard_units(tbl.column(1))
    r = np.mean(first_column_su*second_column_su)
    return r

def fit_line(tbl):
    r = correlation(tbl)
    slope = r * ((np.std(tbl.column(1)))/(np.std(tbl.column(0))))
    intercept = np.mean(tbl.column(1)) - (slope * np.mean(tbl.column(0)))
    return make_array(slope, intercept)

correlation(just_app_and_staffing) # r value for BP staffing and apprehensions

correlation(just_app_and_wall) # r value for wall and apprehensions

resample_slopes = make_array()
for i in np.arange(1000):
    sample = just_app_and_staffing.sample(23)
    resample_line = fit_line(sample)
    resample_slope = resample_line.item(0)
    resample_slopes = np.append(resample_slope, resample_slopes)

Table().with_column("Slope estimate", resample_slopes).hist() # DO NOT CHANGE THIS LINE

lower_end = percentile(2.5, resample_slopes)
upper_end = percentile(97.5, resample_slopes)
print("95% confidence interval for slope: [{:g}, {:g}]".format(lower_end, upper_end))

apprehensions_and_wall_and_staffing.scatter('Staffing', 'Apprehensions Total', fit_line=True)

# Find the predicted apprension total given a staffing number using the regression line
m_and_b = fit_line(just_app_and_staffing) # returns slope and intercept
#Using y = mx+b
staffing_12000 = np.round(m_and_b.item(0)*12000 + m_and_b.item(1))
staffing_12000

just_app_and_staffing
predicted_y = m_and_b.item(0) * just_app_and_staffing.column(0) + m_and_b.item(1)
apprehensions_and_wall_and_staffing.scatter('Staffing', 'Apprehensions Total')
plots.plot(just_app_and_staffing.column(0), predicted_y)

residuals = just_app_and_staffing.column(1) - predicted_y
residuals

rmse = np.sqrt(np.mean(residuals ** 2))
rmse

just_app_and_staffing.select(0).with_column("residuals", residuals).scatter(0)

print("Staffing mean: ", np.std(just_app_and_staffing.select(1)), "Staffing std. deviation: ", 
      np.mean(just_app_and_staffing.select(1)))

