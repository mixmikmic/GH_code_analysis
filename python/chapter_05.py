best_case = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]

worst_case = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def hiring_process(array, cost_interview, cost_hire):
    current_candidate = -1
    cost = 0
    for candidate in array:
        cost += cost_interview
        if candidate > current_candidate:
            current_candidate = candidate
            cost += cost_hire
    return cost

hiring_process(worst_case, 1, 5), hiring_process(best_case, 1, 5)

from math import log, e

def expected_hires(array):
    return log(len(array), e)

len(worst_case) - expected_hires(worst_case)

from numpy import random

def randomize_in_place(array):
    n = len(array)
    iterating_ix = 0
    while iterating_ix < n:
        iterating_value = array[iterating_ix]
        random_ix = random.randint(iterating_ix, n)
        random_value = array[random_ix]
        array[iterating_ix] = random_value
        array[random_ix] = iterating_value
        iterating_ix += 1
    return array

randomize_in_place([1,2,3,4,5])

hiring_process(randomize_in_place(worst_case), 0, 1)

def online_hiring_process(array):
    current_candidate = -1
    ix = 0
    switch_ix = round(len(array)/e)
    while ix <= switch_ix:
        if array[ix] > current_candidate:
            current_candidate = array[ix]
        ix += 1
    while ix < len(array):
        if array[ix] > current_candidate:
            percent_interviewed = ix / len(array) * 100
            return """
            You interviewed {} percent of candidates, and hired {}
            """.format(percent_interviewed, array[ix])
        else:
            ix += 1

for i in range(0,10):
    print(online_hiring_process(randomize_in_place(list(range(0,100)))))

