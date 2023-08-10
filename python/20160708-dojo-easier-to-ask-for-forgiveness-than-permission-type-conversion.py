import csv
import io

def convert(s):
    # First tries to convert input string to an integer.
    # If that does not work, then tries to convert it to a float.
    # If that does not work, leaves it as a string.

    try:
        value = int(s)
    except ValueError:
        pass
    else:
        return value

    try:
        value = float(s)
    except ValueError:
        pass
    else:
        return value

    return s

def convert(s):
    converters = (int, float)
    
    for converter in converters:
        try:
            value = converter(s)
        except ValueError:
            pass
        else:
            return value
        
    return s

data = '''Saeger Buick,123.456,Moosetang
Bobb Ford,234234,Rustang
Mario Fiat,987432.9832,127
'''

print(io.StringIO(data).read(), end='')

with io.StringIO(data) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        items = [convert(s) for s in row]
        print('row: %r becomes:' % row)
        for item in items:
            print('    %r (%s)' % (item, type(item)))    

with io.StringIO(data) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        items = [convert(s) for s in row]
        print('row: {!r} becomes:'.format(row))
        for item in items:
            print('    {0!r} ({1})'.format(item, type(item)))    

# Tolerate (ignore) commas in float numbers.

class EmptyField(str):
    pass

def my_empty_field(s):
    if s:
        raise ValueError
    return EmptyField()  

def my_float(s):
    return float(s.replace(',', ''))

def get_data_converter(s):
    # Note that this returns a converter function,
    # not the converted value.

    converters = (int, my_float, my_empty_field, str)
    
    for converter in converters:
        try:
            converter(s)
        except ValueError:
            pass
        else:
            return converter
        
    assert False, 'Should never get here'

for s in ('19,999.99', 'hello', '1,234', '1234', ''):
    print(repr(s), repr(get_data_converter(s)))

# Is this good?
# I don't know enough about problem to say either way.

def get_data_type(s):
    # Note that this returns type of converted value,
    # not the converter function or converted value.

    converters = (int, my_float, my_empty_field, str)
    
    for converter in converters:
        try:
            value = converter(s)
        except ValueError:
            pass
        else:
            return type(value)
        
    assert False, 'Should never get here'

for s in ('19,999.99', 'hello', '1,234', '1234', ''):
    print(repr(s), repr(get_data_type(s)))

# Could the if/elif/elif of _proc_dict_to_schema_vals()
# be simplified in part with something like the following?

headers = ()  # Stub value to suppress execution and errors.

for field_name in headers:
    types_and_names = (
        # Starts with highest priority,
        # in descending order of priority.
        (str, get_type_of_string(maximum_length)),
        (float, 'Double'),
        (int, 'Long'),
        (EmptyField, get_type_of_string(maximum_length)),
    )
    
    field_types = set(field_types)

    for type_, type_name in types_and_names:
        if type_ in field_types:
            schema.append((field_name, type_name))
            break
    else:
        # Could this be provoked by not having any data rows?
        # If so, which error should be raised?
        raise TypeError('Ran out of types. Should never get here.')

