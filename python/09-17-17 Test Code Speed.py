import cProfile

def slow_function():
    total = 0.0
    for i, _ in enumerate(range(10000)):
        for j, _ in enumerate(range(1,10000)):
            total += (i * j)
    return total

cProfile.run('slow_function()', sort='time')



