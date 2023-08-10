def partition(iterable, left_index, right_index):
    right_value = iterable[right_index]
    min_index = left_index - 1
    for traveling_index in range(left_index, right_index):
        if iterable[traveling_index] <= iterable[right_index]:
            min_index += 1
            min_value = iterable[min_index]
            iterable[min_index] = iterable[traveling_index]
            iterable[traveling_index] = min_value
    pivot_index = min_index + 1
    pivot_value = iterable[right_index]
    iterable[right_index] = iterable[pivot_index]
    iterable[pivot_index] = pivot_value
    return pivot_index

def recursive_quicksort(iterable, left_index, right_index):
    if left_index < right_index:
        print(iterable)
        q = partition(iterable, left_index, right_index)
        recursive_quicksort(iterable, left_index, q-1)
        recursive_quicksort(iterable, q+1, right_index)

def quicksort(iterable):
    left_index = 0
    right_index = len(iterable)-1
    q = partition(iterable, left_index, right_index)
    recursive_quicksort(iterable, left_index, q-1)
    recursive_quicksort(iterable, q+1, right_index)
    return iterable

array = [0,1,9,2,8,3,7,4,6,5]
quicksort(array)

from numpy.random import randint

def random_partition(iterable, left_index, right_index):
    random_integer = randint(left_index, right_index)
    random_value = iterable[random_integer]
    iterable[random_integer] = iterable[right_index]
    iterable[right_index] = random_integer
    return partition(iterable, left_index, right_index)    

def random_quicksort(iterable):
    left_index = 0
    right_index = len(iterable)-1
    q = random_partition(iterable, left_index, right_index)
    recursive_quicksort(iterable, left_index, q-1)
    recursive_quicksort(iterable, q+1, right_index)
    return iterable

array = [9,8,7,6,5,4,3,2,1]
quicksort(array)



