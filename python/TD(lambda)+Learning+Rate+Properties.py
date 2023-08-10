import math

# Which one of these satisfied the above two conditions?
functions = [
    lambda x: 1 / x,
    lambda x: 1 / (x ** 2),
    lambda x: 1 / (x ** (2 / 3)),
    lambda x: 1 / (x ** (1 / 2)),
    lambda x: 1 / 100
]

large_ns = [1000000, 2000000]

values = [
    [
        sum((function(x) for x in range(1, large_n + 1)))
        for function in functions
    ]
    for large_n in large_ns
]

# If a function's sum doesn't grow between two runs
# then it doesn't satisfy the first criteria
for index, (value_large, value_larger) in enumerate(zip(*values)):
    if math.isclose(value_large,
                    value_larger,
                   rel_tol=1e-4):
        print('Function indexed {} stopped growing'.format(index))

values_squared = [
    [
        sum((function(x) ** 2 for x in range(1, large_n + 1)))
        for function in functions
    ]
    for large_n in large_ns
]

# If a function's keeps between two runs
# then it doesn't satisfy the second criteria
for index, (value_large, value_larger) in enumerate(zip(*values_squared)):
    if not math.isclose(value_large,
                    value_larger,
                   rel_tol=1e-4):
        print('Function indexed {} stopped growing'.format(index))

