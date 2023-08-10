import os
import requests

def get_table_url(table_name, base_url=os.environ['NEWSROOMDB_URL']):
    return '{}table/json/{}'.format(os.environ['NEWSROOMDB_URL'], table_name)

def get_table_data(table_name):
    url = get_table_url(table_name)
    
    try:
        r = requests.get(url)
        return r.json()
    except:
        print("Request failed. Probably because the response is huge.  We should fix this.")
        return get_table_data(table_name)

shooting_victims = get_table_data('shootings')

print("Loaded {} shooting victims".format(len(data['shooting_victims'])))

from datetime import date, datetime

def get_shooting_date(shooting_victim):
    return datetime.strptime(shooting_victim['Date'], '%Y-%m-%d')

def shooting_is_this_year(shooting_victim, today):
    try:
        shooting_date = get_shooting_date(shooting_victim)
    except ValueError:
        if shooting_victim['RD Number']:
            msg = "Could not parse date for shooting victim with RD Number {}".format(
                shooting_victim['RD Number'])
        else:
            msg = "Could not parse date for shooting victim with record ID {}".format(
                shooting_victim['_id'])
        
        print(msg)
        return False
        
    return shooting_date.year == today.year

today = date.today()

# Use a list comprehension to filter the shooting victims to ones that
# occured on or before today's month and day.
# Also sort by date because it makes it easier to group by year
shooting_victims_this_year = sorted([sv for sv in shooting_victims
                                       if shooting_is_this_year(sv, today)],
                                      key=get_shooting_date)

import re

def blockify(address):
    """
    Convert a street address to a block level address
    
    Example:
    
    >>> blockify("1440 W 84th St, Chicago, IL 60620")
    '1400 W 84th St, Chicago, IL 60620'
    
    """
    m = re.search(r'^(?P<address_number>\d+) ', address)
    address_number = m.group('address_number')
    block_address_number = (int(address_number) // 100) * 100
    return address.replace(address_number, str(block_address_number))
    

def add_block(sv):
    """Make a copy of a shooting victim record with an added block field"""
    with_block = dict(**sv)
    
    if not sv['Shooting Location']:
        # No location, just set block to none
        print("Record with RD number {0} has no location.".format(
            sv['RD Number']))
        with_block['block'] = None
        return with_block
            
    if sv['Shooting Specificity'] == 'Exact':
        # Address is exact, convert to 100-block
        with_block['block'] = blockify(sv['Shooting Location'])
    else:
        # Address is already block. Use it
        with_block['block'] = sv['Shooting Location']
        
    return with_block

# Create a list of shooting victim dictionaries with blocks
shooting_victims_this_year_with_block = [add_block(sv) for sv in shooting_victims_this_year]

import pandas as pd

# Load shooting victims into a dataframe,
# filtering out victim records for which we couldn't determine the block
shooting_victims_this_year_df = pd.DataFrame([sv for sv in shooting_victims_this_year_with_block if sv['block'] is not None])

# Group by block
shooting_victims_this_year_by_block = shooting_victims_this_year_df.groupby('block').size().sort_values(ascending=False)
shooting_victims_this_year_by_block

# Output to a CSV file so I can email to the reporter who requested it
shooting_victims_this_year_by_block.to_csv("shooting_victims_by_block.csv")



