import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import numpy as np
import copy
import collections

def read_image():
    im = plt.imread('./screencap.png')
    im = im[:, :, :3]
    #     plt.imshow(im)
    return im

# [[0, 1, 1, 1, 1, 1],
#  [0, 1, 1, 1, 1, 0],
#  [2, 2, 1, 0, 0, 1],
#  [0, 0, 1, 0, 1, 1],
#  [1, 1, 1, 1, 1, 1],
#  [1, 1, 1, 0, 0, 0]]

# from sklearn.cluster import KMeans
# import numpy as np

# X = colors
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# kmeans.cluster_centers_

def get_type(color):
    r, g, b = color
        
    empty = [ 0.41390375,  0.28627452,  0.11515152]
    block = [ 0.89002558,  0.48661553,  0.0054561 ]
    red = [ 0.88627452,  0.        ,  0.        ]
    return np.argmin([np.linalg.norm(color - centre) for centre in [empty, block, red]])

def is_wall(line):
    grad = line[1:] - line[:-1] 
    grad_norms = np.sum(grad**2, axis=1)
    return max(grad_norms) > 0.01

#     plt.figure()
#     plt.plot(grad_norms)
#     print grad_norms.shape
#     print np.linalg.norm(grad, ord='fro')

def get_string_rep(grid, wall_grid_x, wall_grid_y):
    string = [['' for _ in range(6)] for _ in range(6)]
    vert_counter = 0
    horiz_counter = 0

    for j in range(6):
        for i in range(6):
            
            if string[j][i]:
                continue
                
            if grid[j][i] == 0:
                string[j][i] = 'e'
            elif grid[j][i] == 2:
                string[j][i] = 'r'

            elif wall_grid_x[j][i]:
                string[j][i] = 'v{}'.format(vert_counter)
                j_temp = j+1
                while not wall_grid_y[j_temp-1][i]:
                    string[j_temp][i] = string[j][i]      
                    j_temp += 1
                vert_counter += 1
            else:
                string[j][i] = 'h{}'.format(horiz_counter)
                i_temp = i+1
                while not wall_grid_x[j][i_temp-1]:
                    string[j][i_temp] = string[j][i]      
                    i_temp += 1
                horiz_counter += 1
                
    return string

def pprint(string):
    for row in string:
        for elem in row:
            print '{:4}'.format(elem),
        print ''
    print ''

def get_moves(string):
    moves = []
    for j in range(6):
        for i in range(6):
            if string[j][i] == 'e':
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                neighbors = [(j + direc[0], i +direc[1]) for direc in directions]
                neighbors = [(y, x) for (y, x) in neighbors if 0<=x<6 and 0<=y<6]
#                 print j, i, neighbors
                dir_map = {(0, 1): ['h', 'r'],
                           (0, -1): ['h', 'r'],
                           (1, 0): 'v',
                           (-1, 0): 'v',                           
                          }
                for (y, x) in neighbors:
                    direc = (j-y, i-x)
                    if string[y][x][0] in dir_map[direc]:
                        move = (j, i, direc)
                        moves.append(move)
                        
    return moves
                        
    
def apply_move(old_string, move):
    string = copy.deepcopy(old_string)
    
    j, i, direc = move
    dy, dx = direc
    
    label = string[j-dy][i-dx]
    
    j = j - dy
    i = i - dx
    while 0<=j<6 and 0<=i<6 and string[j][i] == label:
        string[j][i] = 'e'
        string[j+dy][i+dx] = label
        j = j - dy
        i = i - dx
        
    return string


def get_neighbors(string):
    moves = get_moves(string)
    neighbors = [apply_move(string, move) for move in moves]
    return neighbors, moves

def hashed(string):
    return ''.join([' '.join(row) for row in string])

def check_solved(string):
    return (string[2][4] == 'r' and string[2][5] == 'r')

def bfs(start):
    visited = {}
    back_pointer = {}
    queue = collections.deque([start])
    visited[hashed(start)] = True
    
    while len(queue) > 0:
        node = queue.popleft()
        neighbors, moves = get_neighbors(node)
        for neighbor, move in zip(neighbors, moves):
            if check_solved(neighbor):
                back_pointer[hashed(neighbor)] = node, move
                return back_pointer, neighbor
                
            if hashed(neighbor) in visited:
                continue
            else:
                queue.append(neighbor)
                visited[hashed(neighbor)] = True 
                back_pointer[hashed(neighbor)] = node, move
                
    return False, None

def get_path(back_pointer, start, final):
    path = [(final, None)]
    while path[-1][0] != start:
        prev_node, prev_move = path[-1]
        node, move = back_pointer[hashed(prev_node)]
        path.append((node, move))
    return list(reversed(path))

def get_swipes(path):
    swipes = []
    for node, move in path:
        x, y, direc = move
        dx, dy = direc
        x1, y1 = x - dx, y - dy
        x2, y2 = x, y
        swipe = (x1, y1, x2, y2)
        swipes.append(swipe)
    return swipes

im = read_image()
 
X = [127, 294, 461, 628, 795, 962]
Y = [633, 800, 967, 1134, 1301, 1469]

grid =[[None for _ in range(6)] for _ in range(6)]

colors = []
for i, y in enumerate(Y):
    for j, x in enumerate(X):
        color = im[y, x, :]
        grid[i][j] = get_type(color)
        colors.append(color)
        
wall_grid_x = [[True for i in range(6)] for j in range(6)]
wall_grid_y = [[True for i in range(6)] for j in range(6)]

for j in range(6):
    for i in range(5):
        line = im[Y[j], X[i]:X[i+1], :]
        wall_grid_x[j][i] = is_wall(line)

for j in range(5):
    for i in range(6):
        line = im[Y[j]:Y[j+1], X[i], :]
        wall_grid_y[j][i] = is_wall(line)       

string = get_string_rep(grid, wall_grid_x, wall_grid_y)
# pprint(string)

neighbors = get_neighbors(string)

# for neighbor in neighbors:
#     pprint(neighbor)
#     print ''

back_pointer, final = bfs(string)

start = string
path = get_path(back_pointer, start, final)
swipes = get_swipes(path[:-1])
swipes.append((2, 4, 2, 5))

swipes

