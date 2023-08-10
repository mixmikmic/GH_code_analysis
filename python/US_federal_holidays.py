from datetime import date
import holidays
import csv
import sys

us_holidays = holidays.UnitedStates() 

# writes a new csv file with holidays recognized in OR using the holidays module

with open('or_holidays.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['date', 'holiday'])
    for date, name in sorted(holidays.US(state='OR', years=[2010, 2011, 2012, 2013, 2014, 2015, 2016]).items()):
        print(date, name)
        writer.writerow([date, name])
            



