import numpy as np
from __future__ import print_function

empty = 0
x = 1
oh = -1
init_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

class Node:
    def __init__(self, board, depth):
        self.board = board
        self.prob = 0.5
        self.children = {}   # key = (player, (x,y)) , value = node
        self.depth = depth
        
    def __repr__(self):
        return '\n'+ '\t'*self.depth + str(self.board).replace('\n', '\n'+'\t'*self.depth) +" "+str(self.prob)+' '+str(self.children.values())

root = Node(init_board, 0) 
print(root)

def expand(node, position, player):
    if node.board[position] != 0:
        key = (node.board[position], position)
        return None
    key = (player, position)
    new_board = np.copy(node.board)
    new_board[position] = player
    new_child = Node(new_board, node.depth+1)
    node.children[key] = new_child
    return new_child

expand(root,(0,0),x)
select = expand(root,(1,1),oh)
expand(select, (2,2),x)
expand(select, (2,2),oh)
expand(select, (1,1),x)
print(root)

board = root.board
rows = len(board)
cols = len(board[0])
def Winner(board):
    for i in range(0,rows):
        row = board[i,:]
        if sum(row) == x*cols:
            return x
        elif sum(row) == oh*cols:
            return oh
    for j in range(0, cols):
        col = board[:,j]
        if sum(col) == x*rows:
            return x
        elif sum(col) == oh*rows:
            return oh
    diag_right = board.diagonal()
    diag_left = board[:, ::-1].diagonal()
    if sum(diag_right) == x*rows:
        return x
    if sum(diag_left) == x*rows:
        return x
    if sum(diag_right) == oh*rows:
        return oh
    if sum(diag_left) == oh*rows:
        return oh
    return 0
print(Winner(np.array([[x,x,x], [0,0,0], [0,oh,oh]])))
print(Winner(np.array([[x,oh,0], [x,0,0], [x,oh,oh]])))
print(Winner(np.array([[x,oh,0], [x,oh,0], [oh,oh,0]])))
print(Winner(np.array([[x,oh,0], [x,x,0], [oh,oh,x]])))
print(Winner(np.array([[x,oh,oh], [x,oh,0], [oh,0,0]])))
print(Winner(init_board))

def solve(root, next_player):
    winner = Winner(root.board)
    if winner != 0:
        root.prob = 0.0 if winner == oh else 1.0
        return
    root.prob = 0.0
    for i in range(len(board)):
        for j in range(len(board[0])):
            expand(root, (i,j), next_player)
    if len(root.children) == 0:
        return
    for child in root.children.values():
        solve(child, next_player * -1)
        root.prob += child.prob
    root.prob /= len(root.children)

root = Node(init_board, 0)
solve(root, x)

def play_x_best(node):
    return max(node.children.values(), key= lambda x: x.prob)
def play_x_sample(node):
    p=[float(x.prob) for x in node.children.values()]
    p_norm = [x / sum(p) for x in p]
    return np.random.choice(node.children.values(), p=p_norm)
def play_oh(node):
    return min(node.children.values(), key= lambda x: x.prob)
play_x_best(play_x_best(play_x_best(root).children.values()[0]).children.values()[0])

from ipywidgets import interactive, Button, RadioButtons, widgets
from IPython.display import display
import ipywidgets, IPython.display



node = root
buttons = []

def create_board():
    for i in range(0,len(board)):
        button_row = []
        for j in range(0,len(board[i])):
            b = Button()
            b.on_click(play_game)
            button_row.append(b)
        buttons.append(button_row)
        display(widgets.Box(button_row))
    update_board()
    
def update_board():
    for i in range(0,len(board)):
        for j in range(0,len(board[i])):
            pos = (i,j)
            player = node.board[pos]
            b = buttons[i][j]
            b.description='x' if player == x else 'o' if player == oh else '.'
            b.board_position = pos

def play_game(button):
    global node
    point = button.board_position
    # your move
    node = node.children[(oh, point)]
    # computer's move
    node = play_x_best(node)      # Change this to play_x_sample to have the opponent sample randomly from the best moves.
    update_board()

create_board()
node = play_x_sample(node)       # Start with random good move.
update_board()

print("Run this cell!")





