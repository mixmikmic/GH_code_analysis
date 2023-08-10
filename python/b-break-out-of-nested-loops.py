N_COLUMNS = 5
N_LINES = 3

import numpy as np

a = np.arange(N_LINES * N_COLUMNS).reshape(N_LINES, N_COLUMNS)
a

please_break = False
for line in range(N_LINES):
    for column in range(N_COLUMNS):
        if a[line, column] == 8:
            please_break = True
            break
        a[line, column] = -1
    if please_break:
        break
        
print(a)    

a = np.arange(N_LINES * N_COLUMNS).reshape(N_LINES, N_COLUMNS)
a

def get_array_indexes(n_rows, n_columns):
    for row in range(n_rows):
        for column in range(n_columns):
            yield row, column

for line, column in get_array_indexes(N_LINES, N_COLUMNS):
    if a[line, column] == 6:
        break
    a[line, column] = -1

print(a)

a = np.arange(N_LINES * N_COLUMNS).reshape(N_LINES, N_COLUMNS)
a

def foo(a, n_lines, n_columns):
    for line in range(n_lines):
        for column in range(n_columns):
            if a[line, column] == 7:
                return a
            a[line, column] = -1
    return a

foo(a, N_LINES, N_COLUMNS)

