MONTHS_PER_YEAR = 12

# for unknown year
def max_month_length(month):
    """Return maximum number of days for given month.
    month is zero-based.
    That is,
    0 means January,
    11 means December,
    12 means January (again)
    -2 means November (yup, wraps around both ways)"""
    max_month_lengths = (
        31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return max_month_lengths[month % len(max_month_lengths)]

max_month_length(0)  # January

max_month_length(13)  # February

max_month_length(-2)  # November

def max_days_n_months(n_months, starting_month=None):
    """Return the maximum number of days
    in n_months whole consecutive months,
    optionally starting with starting_month.
    
    If starting_month is None or is not specified,
    return highest value for all possible starting months."""
    if starting_month is not None:
        return sum(
            max_month_length(month)
            for month in range(starting_month, starting_month + n_months)
        )
    
    return max(
        max_days_n_months(n_months, starting_month)
        for starting_month in range(MONTHS_PER_YEAR)
    )

def foo(n):
    for n_months in range(1, n+1):
        n_days = max_days_n_months(n_months)
        yield n_months, n_days

n = MONTHS_PER_YEAR
# %timeit list(foo(n))
list(foo(n))

from collections import defaultdict

def max_n_days_for_months():
    """Yield tuples of
        number of consecutive months,
        maximum number of days for those consecutive months
        and list of starting months which produce that above maximum
    for all numbers of consecutive months up to a year."""
    for n_months in range(1, MONTHS_PER_YEAR+1):
        d = defaultdict(list)
        for starting_month in range(MONTHS_PER_YEAR):
            n_days = max_days_n_months(n_months, starting_month)
            d[n_days].append(starting_month)
        max_n_days = max(d)
        yield n_months, max_n_days, sorted(d[max_n_days])

# %timeit list(max_n_days_for_months())
list(max_n_days_for_months())

def pretty():
    """Yields lines to be absolutely identical 
    to that from days_spanned.sh."""
    for selector, name in ((min, '-gt'), (max, '-ge')):
        yield f'Compare: {name} '
        for n_months, max_n_days, months in max_n_days_for_months():
            month = selector(months) + 1
            if n_months == 1:
                yield f'month: {month} month spans: {max_n_days} days  '
            elif n_months == 2:
                yield f'month: {month} plus following month spans: {max_n_days} days  '
            else:
                yield f'month: {month} plus following {n_months - 1} months spans: {max_n_days} days  '
        yield ''

for line in pretty():
    print(line)

known_good_output = ''.join(f'{line}\n' for line in pretty())

def pretty():
    """Yields lines to be absolutely identical 
    to that from days_spanned.sh."""
    for selector, name in ((min, '-gt'), (max, '-ge')):
        yield f'Compare: {name} '
        for n_months, max_n_days, months in max_n_days_for_months():
            month = selector(months) + 1
            if n_months == 1:
                duration_prose = f'month'
            elif n_months == 2:
                duration_prose = f'plus following month'
            else:
                duration_prose = f'plus following {n_months - 1} months'
            yield f'month: {month} {duration_prose} spans: {max_n_days} days  '
        yield ''

assert known_good_output == ''.join(f'{line}\n' for line in pretty())

MONTH_NAMES = '''
    January February March April May June
    July August September October November December
'''.split()

MONTH_NAMES

# derived from https://stackoverflow.com/questions/38981302/converting-a-list-into-comma-separated-string-with-and-before-the-last-item

def oxford_comma_join(items, join_word='and'):
    # print(f'items={items!r} join_word={join_word!r}')
    items = list(items)
    if not items:
        return ''
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f' {join_word} '.join(items)
    else:
        return ', '.join(items[:-1]) + f', {join_word} ' + items[-1]

test_data = (
    # (args for oxford_comma_join, correct output),
    ((('',),), ''),
    ((('lonesome term',),), 'lonesome term'),
    ((('here', 'there'),), 'here and there'),
    ((('you', 'me', 'I'), 'or'), 'you, me, or I'),
    ((['here', 'there', 'everywhere'], 'or'), 'here, there, or everywhere'),
)

for args, known_good_output in test_data:
    # print(f'args={args!r}, k={known_good_output!r}, output={oxford_comma_join(*args)!r}')
    assert oxford_comma_join(*args) == known_good_output

import inflect

p = inflect.engine()

from textwrap import wrap

def pretty():
    """For number of consecutive months, up to 12,
    yields sentences that show for each number of consecutive months,
    the maximum possible number of days in those consecutive months,
    and for which starting months one can have
    those maximum possible number of days."""
    for n_months, max_n_days, months in max_n_days_for_months():
        month_names = (MONTH_NAMES[month] for month in months)
        yield (
            f'{n_months} consecutive {p.plural("month", n_months)} '
            f'can have at most {max_n_days} days '
            f'if starting in {oxford_comma_join(month_names, "or")}.'
        )

for sentence in pretty():
    for line in wrap(sentence):
        print(line)

def not_so_pretty():
    """For number of consecutive months, up to 12,
    yields sentences that show for each number of consecutive months,
    the maximum possible number of days in those consecutive months,
    and for which starting months one can have
    those maximum possible number of days."""
    for n_months, max_n_days, months in max_n_days_for_months():
        month_names = (MONTH_NAMES[month][:3] for month in months)
        yield (
            f'{n_months} months '
            f'have max {max_n_days} days '
            f'starting in {oxford_comma_join(month_names, "or")}.'
        )

for sentence in not_so_pretty():
    print(sentence)

