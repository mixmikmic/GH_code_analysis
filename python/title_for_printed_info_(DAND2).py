def get_city():
    '''Asks the user for one of three cities and returns the city name and filename for 
       that city's bike share data.

    Args:
        none.
    Returns:
        (str) lowercase name of city, (str) filename for a city's bikeshare data from input.
    '''
    # Ask user for input while managing incorrect input
    while True:
        city = input('\nHello! Let\'s explore some US bikeshare data!\n'
                     'Would you like to see data for Chicago, New York, or Washington?\n')
        # Confirm if input string is one of the listed cities and ask again if not
        if city.lower() not in ('chicago', 'new york', 'washington'):
            print('\nYou didn\'t enter the correct input. Please enter one of the cities listed.\n'
                  'Returning you to the original input request:')
        else:
            break
    # Use the input to select a filename
    city = city.lower()
    if city == 'chicago':
        file = 'chicago.csv'
    elif city == 'new york':
        file =  'new_york_city.csv'
    else:
        file = 'washington.csv'
    return city, file

city = ('chicago', 'chicago.csv')

print(city[1])

print('\n--- Printing US Bikeshare Statistics for', city[0].title(), '---')

print('\n--- Printing US Bikeshare Statistics for', city[0].title(), '---\n'
      '    (Full city data provided)')

selected_month = 'January'

print('\n--- Printing US Bikeshare Statistics for', city[0].title(), '---\n'
      '    (Data for', selected_month,'provided)')

from calendar import day_name

selected_day = 1

print('\n--- Printing US Bikeshare Statistics for', city[0].title(), '---\n'
      '    (Data for all {}s provided)'.format(day_name[selected_day]))

print('\n--- Printing US Bikeshare Statistics for', city[0].title(), '---\n'
      '    (Data for all {}s in {} provided)'.format(day_name[selected_day], selected_month))



