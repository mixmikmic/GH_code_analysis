year, month, day = 'hello', -1, 0
date_formats = {
    'iso': f'{year}-{month:02d}-{day:02d}',
    'us': f'{month}/{day}/{year}',
    'other': f'{day} {month} {year}',
}

year, month, day = 2017, 3, 27
print(year, month, day)
print(date_formats['iso'])

year, month, day = 'hello', -1, 0
# year, month, and day do not have to be defined when creating dictionary.
del year  # Test that with one of them.
date_formats = {
    'iso': (lambda: f'{year}-{month:02d}-{day:02d}'),
    'us': (lambda: f'{month}/{day}/{year}'),
    'other': (lambda: f'{day}.{month}.{year}'),
}
dates = (
    (2017, 3, 27),
    (2017, 4, 24),
    (2017, 5, 22),
)

for format_name, format in date_formats.items():
    print(f'{format_name}:')
    for year, month, day in dates:
        print(format())

