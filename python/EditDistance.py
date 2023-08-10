get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

def print_matrix(m):
    for i, row in enumerate(m):
        print ', '.join(map(str, row))

def edit_distance_matrix(s, t):

    # add a special "start" character to beginning of each string
    s = '_' + s
    t = '_' + t

    d = [ [None] * len(t) for i in range(len(s)) ]
    
    for i in range(len(s)):
        d[i][0] = i
    for j in range(len(t)):
        d[0][j] = j

    for i in range(1, len(s)):
        for j in range(1, len(t)):

            # sub/copy: s[i] is "replaced" with t[j] if s[i] == t[j], this is 
            #           simply copying over s[i] and adds no cost
            if s[i] == t[j]:
                cost_sub = d[i-1][j-1] + 0
            else:
                cost_sub = d[i-1][j-1] + 1

            # insert: t[j] is inserted at end of s[:i].  the cost is
            #         cost of insert + cost of transforming s[:i] to t[:j-1]
            cost_insert = 1 + d[i][j-1]

            # delete: s[i] is deleted at end of s[:i].  the cost is
            #         cost of delete + cost of transforming s[:i-1] to t[:j]
            cost_delete = 1 + d[i-1][j]

            d[i][j] = min(cost_sub, cost_insert, cost_delete)

    return d

print_matrix(edit_distance_matrix('xyzABC', 'ABC'))  # del,del,del,copy,copy,copy

print_matrix(edit_distance_matrix('ABC', 'ABCxyz'))  # copy,copy,copy,ins,ins,ins

print_matrix(edit_distance_matrix('BCD', 'ABC'))     # ins, copy, copy, del



