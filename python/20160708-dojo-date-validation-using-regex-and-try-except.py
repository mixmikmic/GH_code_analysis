import re
import datetime

date_pattern = re.compile(r'''
    # Expects dates to be in YYYY?MM?DD format
    # where ? is separator.
    # The separator may be one of '-', ',', '.', '/', or ' '.
    # Both separators must be the same.
    # Month and day may each have either one or two digits.
    
    # The (?P<name>...) stuff is fantastic for allowing one
    # to pull things out of a match object by names.
    
    ^                         # matches beginning of string
    (?P<year>\d{4})           # year must have exactly four digits
    (?P<separator>[-,./ ])    # separator must be one of
                              # '-', ',', '.', '/', or ' '.
    (?P<month>\d{1,2})        # month must have one or two digits
    (?P=separator)            # must match earlier separator
    (?P<day>\d{1,2})          # day must have one or two digits
    $                         # matches end of string
''', re.VERBOSE)

min_date = datetime.date(1900, 1, 1)
max_date = datetime.date(2100, 1, 1) - datetime.timedelta(days=1)

sample_input = [
    '2016-7-6',
    '2016-07-06',
    '2016 07-06',
    '2016 07 06',
    '2016/07 06',
    '2016/07/06',
    '2016-07/06',
    '2016-02-29',
    '2015-02-29',
    '1899-12-31',
    '1900-01-01',
    '2099-12-31',
    '2100-01-01',
]

# This shows the input and output
# for applying the date_pattern regex to sample_input.
# Note that 2015-02-29 passes the regex,
# although it is not a valid date.

for s in sample_input:
    print('%r ' % s, end='')
    m = re.match(date_pattern, s)
    if not m:
        print('no match')
        continue
    print(
        'year=%r, month=%r, day=%r separator=%r' % (
        m.group('year'),
        m.group('month'),
        m.group('day'),
        m.group('separator'),
    ))

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    try:
        date = datetime.date(
            int(m.group('year')),
            int(m.group('month')),
            int(m.group('day')),
        )
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

for s in sample_input:
    date = convert_to_date(s)
    print('%r -> %s' % (s, date))

presumed_good_output = [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
#
# It minimizes the code inside the try clause
# by getting the year, month, and day strings from the match object
# outside the try clause.
# It is also easy to read.

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year  = int(m.group('year'))
    month = int(m.group('month'))
    day   = int(m.group('day'))
    try:
        date = datetime.date(year, month, day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
#
# The year, month, and day are gotten as a tuple
# from the match object in a single expression.
# It is correct but hard to read,
# so the previous version of the function is better.

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year_month_day = map(
        lambda name: int(m.group(name)),
        ('year', 'month', 'day')
    )
    try:
        date = datetime.date(*year_month_day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
#
# match.group() can handle multiple names, returning a tuple.
# The year_month_day stuff is now easier to read.
# How does this code compare to two cells ago?
# Which is easier to understand?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year_month_day = map(int, m.group('year', 'month', 'day'))
    try:
        date = datetime.date(*year_month_day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
#
# Went back to separate year, month, and day variables.

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year, month, day = map(int, m.group('year', 'month', 'day'))
    try:
        date = datetime.date(year, month, day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

date_pattern = re.compile(r'''
    # Expects dates to be in YYYY?MM?DD format
    # where ? is separator.
    # The separator may be one of '-', ',', '.', '/', or ' '.
    # Both separators must be the same.
    # Month and day may each have either one or two digits.
    
    # Stuff in parentheses can be gotten from the match object
    # by index.
    
    ^            # beginning of string
    (\d{4})      # group 1: year must have exactly four digits
    ([-,./ ])    # group 2: separator must be one of
                 # '-', ',', '.', '/', or ' '.
    (\d{1,2})    # group 3: month must have one or two digits
    \2           # must match earlier separator (of group 2)
    (\d{1,2})    # group 4: day must have one or two digits
    $            # end of string
''', re.VERBOSE)

# The function in this cell works the same as the earlier version.

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year  = int(m.group(1))
    month = int(m.group(3))
    day   = int(m.group(4))
    try:
        date = datetime.date(year, month, day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
# It is a minor variation on the above,
# consolidating three statements:
#
#     year  = int(m.group(1))
#     month = int(m.group(3))
#     day   = int(m.group(4))
#
# into one statement:
#
#     year, month, day = map(int, m.group(1, 3, 4))
#
# Which way is is easier to read?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year, month, day = map(int, m.group(1, 3, 4))
    try:
        date = datetime.date(year, month, day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
#
# It is a minor variation on the above,
# getting all four groups at once.
# The name _ is used for the unwanted group.
# Using the name _ for unwanted data, comes from Ruby.
# Converting the strings to integers has to be done
# in a separate step to avoid crashing while 
# converting a separator to an int.
#
# What benefits does this technique have?
# What drawbacks does this technique have?
#
# Is this easier to read than the previous cell?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year, _, month, day = m.groups()
    year, month, day = map(int, (year, month, day))
    try:
        date = datetime.date(year, month, day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
# It is a minor variation on the function two cells above,
# consolidating three variables:
#
#     year, month, day
#
# into a single sequence:
#
#     year_month_day
#
# Note the *year_month_day argument to datetime.date().
#
# Which way is more readable?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    year_month_day = map(int, m.group(1, 3, 4))
    try:
        date = datetime.date(*year_month_day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

date_pattern = re.compile(r'''
    # Expects dates to be in YYYY?MM?DD format
    # where ? is separator.
    # The separator may be one of '-', ',', '.', '/', or ' '.
    # Both separators must be the same.
    # Month and day may each have either one or two digits.
    
    ^            # beginning of string
    \d{4}        # year must have exactly four digits
    ([-,./ ])    # group 1: separator must be one of
                 # '-', ',', '.', '/', or ' '.
    \d{1,2}      # month must have one or two digits
    \1           # must match earlier separator (of group 1)
    \d{1,2}      # day must have one or two digits
    $            # end of string
''', re.VERBOSE)

# The function in this cell works the same as the earlier version.
# It gets the separator from the match object,
# then uses that to split the string into year_month_day list.
#
# Which way is more readable?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    separator = m.group(1)
    year_month_day = map(int, s.split(separator))
    try:
        date = datetime.date(*year_month_day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

# The function in this cell works the same as the earlier version.
# It does not use the regex for parsing.
#
# Which way is more readable?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    # Convert the non-digits to spaces, for easy splitting.
    s = ''.join([
        c if c.isdigit() else ' '
        for c in s
    ])
    year_month_day = map(int, s.split())
    try:
        date = datetime.date(*year_month_day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

date_pattern = re.compile(r'''
    # Expects dates to be in YYYY?MM?DD format
    # where ? is separator.
    # The separator may be one of '-', ',', '.', '/', or ' '.
    # Both separators must be the same.
    # Month and day may each have either one or two digits.
    
    ^            # beginning of string
    (\d{4}-\d{1,2}-\d{1,2})
    |
    (\d{4},\d{1,2},\d{1,2})
    |
    (\d{4}\.\d{1,2}\.\d{1,2})
    |
    (\d{4}/\d{1,2}/\d{1,2})
    |
    (\d{4}\ \d{1,2}\ \d{1,2})
    $            # end of string
''', re.VERBOSE)

# The function in this cell works the same as the earlier version.
# It does not use the regex for parsing.
#
# Which way is more readable? This way or the previous way?

def convert_to_date(s):
    '''Converts input string s to a datetime.date object.
    
    Returns datetime.date object if s is valid date.
    Date must be between min_date and max_date inclusive.
    Otherwise returns None.'''
    
    m = re.match(date_pattern, s)
    if not m:
        return None
    
    # Convert the non-digits to spaces, for easy splitting.
    s = ''.join([
        c if c.isdigit() else ' '
        for c in s
    ])
    year_month_day = map(int, s.split())
    try:
        date = datetime.date(*year_month_day)
    except ValueError:
        return None

    if min_date <= date <= max_date:
        return date
    else:
        return None

assert presumed_good_output == [convert_to_date(s) for s in sample_input]

