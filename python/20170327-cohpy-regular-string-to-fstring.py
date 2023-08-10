date_formats = {
    'iso': '{year}-{month:02d}-{day:02d}',
    'us': '{month}/{day}/{year}',
    'other': '{day} {month} {year}',
}
dates = (
    (2017, 3, 27),
    (2017, 4, 24),
    (2017, 5, 22),
)

for format_name, format in date_formats.items():
    print(f'{format_name}:')
    for year, month, day in dates:
        print(eval('f%r' % format))

