#Author:  Joe Friedrich
#
#Important Note!!  PYTHON 3 ONLY!

import datetime
import calendar
import itertools

def get_month_from_user():
    while True:
        try:
            input_month = int(input('Please enter the calendar month number:  '))
            if 0 < input_month < 13: 
                return input_month
            print ("The number entered does not correspond with a month on the calendar.\n"
                   'Please try again.')
        except ValueError:
            print ('That was not a whole number.  Please try again.')

def find_december_monday(currentYear):
    month = 12
    dates = []

    for year in range(2009, currentYear + 1):
        day = 1
    
        if calendar.weekday(year, month, day) == calendar.MONDAY:
            day += 7
            dates.append(str(month) + '/' + str(day) + '/' + str(year)) 
        elif calendar.weekday(year, month, day + 1) == calendar.MONDAY:
            day += 8
            dates.append(str(month) + '/' + str(day) + '/' + str(year))
        else:
            for day in range(3, 8):
                if calendar.weekday(year, month, day) == calendar.MONDAY:
                    dates.append(str(month) + '/' + str(day) + '/' + str(year))

    return dates

def find_may_monday(currentYear):
    month = 5
    dates = []

    for year in range(2010, currentYear + 1):
        day = calendar.monthrange(year, month)[1]
    
        for day in range(day, 1, -1):
            if calendar.weekday(year, month, day) == calendar.MONDAY:
                day -= 7
                dates.append(str(month) + '/' + str(day) + '/' + str(year))
                break

    return dates

def find_last_monday(currentYear, month):
    dates = []
    
    if month == 9 or month == 10:
        startYear = 2009
    else:
        startYear = 2010
    
    for year in range(startYear, currentYear + 1):
        day = calendar.monthrange(year, month)[1]
    
        for day in range(day, 1, -1):
            if calendar.weekday(year, month, day) == calendar.MONDAY:
                dates.append(str(month) + '/' + str(day) + '/' + str(year))
                break

    return dates

currentYear = datetime.date.today().year
month = get_month_from_user()

if month == 11:
    print ('COhPy does not meet in November.')
elif month == 12:
    print (find_december_monday(currentYear))
elif month == 5:
    print (find_may_monday(currentYear))
else:
    print (find_last_monday(currentYear, month))



