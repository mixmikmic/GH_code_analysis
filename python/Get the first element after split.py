def get_first_split(input_string, split_by=' '):
    try:
        input_string_stripped = input_string.strip()
        split_by_index = input_string_stripped.find(split_by)
        if split_by_index == -1:
            return input_string_stripped
        else:
            substring = input_string_stripped[:split_by_index].strip()
            return substring
    except AttributeError:
        return ''

get_first_split(' this is  _ a test', '_')

get_first_split('some_group_1', '_')

x = 'XQWRQW' * 100000
x += ' ' + ('XWQQWFW' * 100000)

len(x)

get_ipython().run_cell_magic('timeit', '', 'x.split()[0]')

get_ipython().run_cell_magic('timeit', '', 'x.split(maxsplit=1)[0]')

get_ipython().run_cell_magic('timeit', '', 'get_first_split(x)')

