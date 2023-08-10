width = 1000

def foo(width):
    horizontal_line = '{'
    for i in range(width):
        horizontal_line += (' ' + '#ffffff')
    horizontal_line += '}'

get_ipython().magic('timeit foo(width)')

def foo(width):
    horizontal_line = (
        '{' +
        ' '.join(
            '#ffffff' for i in range(width)
        ) +
        '}'
    )

get_ipython().magic('timeit foo(width)')

def foo(width):
    horizontal_line = ''.join([
        '{',
        ' '.join(
            '#ffffff' for i in range(width)
        ),
        '}',
    ])

get_ipython().magic('timeit foo(width)')

