import random

# Binary search

#     0  1  2  3  4  5  6   7
ls = [1, 2, 3, 6, 7, 9, 13, 16]

def binary_search(elem, ls):
    """
    Search for the given elem in the given list and return the index.
    Return -1 is the elem was not found.
    """
    low = 0
    high = len(ls)-1
    while True:
        middle = low + ((high - low) // 2)
        if ls[middle] == elem:
            return middle
        if low == high:
            return -1
        elif ls[middle] > elem:
            high = middle - 1
        elif ls[middle] < elem:
            low = middle + 1

print(binary_search(4, ls))
print([binary_search(i, ls) for i in ls])

# Quicksort

def quicksort(ls):
    """
    Sort the given list inplace.
    """
    quicksort_rec(ls, 0, len(ls)-1)

def quicksort_rec(ls, start, stop):
    """
    Sort the given list between start and stop inplace
    """
    if start >= stop:
        return
    else:
        # partition
        part_idx = partition(ls, start, stop)
        # sort left and right
        quicksort_rec(ls, start, part_idx)
        quicksort_rec(ls, part_idx+1, stop)
    
def partition(ls, start, stop):
    """
    parition the given list between start and stop, so that all elements smaller 
    than ls[start] are to the left of ls[start] and all bigger are to the right 
    of ls[start].
    Return the partition index
    """
    part_elem = ls[start]
    idx_left = start + 1
    idx_right = stop
    while idx_left < idx_right:
        if ls[idx_left] > part_elem:
            if ls[idx_right] > part_elem:
                # If the element on the left needs to be moved to the right
                # but the right element needs to stay right,
                # get the next right element.
                idx_right -= 1
            else:
                # ls[idx_right] <= part_elem
                # If the element on the left needs to be moved to the right
                # and the right element needs to be moved to the left: swap and update both indices
                ls[idx_left], ls[idx_right] = ls[idx_right], ls[idx_left]
                idx_left += 1
                idx_right -= 1
        else:
            # ls[idx_left] <= part_elem
            # Index on the left is at right place, update index
            idx_left += 1
    if ls[idx_left] > part_elem:
        idx_left -= 1
    # move the current elemnt at the previous smaller index to the beginning
    ls[start] = ls[idx_left]
    ls[idx_left] = part_elem
    return idx_left
            
ls = random.sample(range(1, 10), 8)
print(ls)
quicksort(ls)
print(ls)

