def swap(a, b):
    b, a = a, b
    return a, b

print swap(4,10)

grid = [['X', 'O', 'X'],
        ['O', 'X', 'O'],
        ['O', 'X', 'O']]

def tic_tac_toe_win(grid):
    ## Check across for wins
    across = [''.join(row) for row in grid]
    if 'XXX' in across or 'OOO' in across:
        return True
    
    ## Check down
    down = [''.join([r[c] for r in grid]) for c in range(3)]
    if 'XXX' in down or 'OOO' in down:
        return True
    
    ## Check diagonals
    diagonal = [''.join([grid[p][p] for p in range(3)]),
                ''.join([grid[p][len(grid)-p-1] for p in range(3)])]
    if 'XXX' in diagonal or 'OOO' in diagonal:
        return True
    
    return False
    
print tic_tac_toe_win(grid)

## 20! = product[1,2,3,4,5,6,7,8,9,10,11...20]
import math

def trailing_zeroes(n):
    multiplers = range(1, n+1)
    k = int(math.log(n, 5))    
    return sum([n/(5**x) for x in range(1, k+1)])
    
trailing_zeroes(500)

def maximum_of_two(x, y):
    return max(x, y)

maximum_of_two(45, 12)

def mastermind(guess, answer):
    ## Matching the exact positions for hits
    match = [i for i, j in zip(guess, answer) if i == j]
    hits = len(match)

    if len(match) == 4:
        print 'you win!'
        return

    ## Remove any hits and check remaining for number of pseudo-hits
    unguessed = [x for x in answer if x not in match]
    other_guesses = [x for x in guess if x not in match]

    pseudo = len(set(unguessed).intersection(set(other_guesses)))

#     print answer
    print 'Your guess:', guess
    print 'Hits:', hits, '\tPseudo-hits:', pseudo

mastermind(guess = 'rggy', answer = 'rbgy')

def sublist_sort(x):
    for index, num in enumerate(x):
        if num > min(x[index:]):
            m = index
            break

    for index, num in enumerate(x):
        print x[:index], num
#         if num > max(x[:index])

    print 'm =', m
    print 'n =', n
    
    return m, n

x = [1, 2, 4, 7, 10, 7, 12, 6, 7, 16, 18, 19]
sublist_sort(x)

# sublist_sort([1,5,4,12,10, 20])





def biggest_block(string):
    """
    Inchworm through the block, calculating each sum while incrementing end index.
    But if the sum is less than 0, it won't help any subsequent blocks, so we
    reset by changing position of the start index.
    """
    start = 0
    end = 1
    biggest_block = [start, end]
    biggest_sum = None
    n = len(string)
    while end < n:
        new_sum = sum(string[start:end])
        if new_sum > biggest_sum:
            biggest_sum = new_sum
            biggest_block = [start, end]
        if new_sum < 0:
            start = end
        end += 1
    return string[biggest_block[0]:biggest_block[1]]

# Test Cases
print biggest_block([7, 9, -3, -10, -7, 8, -5, -5, -7, -7, 5, 1, 5, -8, 0, 8, -2, 4, 2, -1])
# [7, 9]

print biggest_block([-2, 1, 2, 3, -1, 0, 100, -101, 50])
# [1, 2, 3, -1, 0, 100]

print biggest_block([2, -8, 3, -2, 4, -10])
# [3, -2, 4]



