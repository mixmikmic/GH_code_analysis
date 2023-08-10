import time

class Problem(object):

    def __init__(self, initial):
        self.initial = initial
        self.type = len(initial) # Defines board type, either 6x6 or 9x9
        self.height = int(self.type/3) # Defines height of quadrant (2 for 6x6, 3 for 9x9)

    def goal_test(self, state):
        # Expected sum of each row, column or quadrant.
        total = sum(range(1, self.type+1))

        # Check rows and columns and return false if total is invalid
        for row in range(self.type):
            if (len(state[row]) != self.type) or (sum(state[row]) != total):
                return False

            column_total = 0
            for column in range(self.type):
                column_total += state[column][row]

            if (column_total != total):
                return False

        # Check quadrants and return false if total is invalid
        for column in range(0,self.type,3):
            for row in range(0,self.type,self.height):

                block_total = 0
                for block_row in range(0,self.height):
                    for block_column in range(0,3):
                        block_total += state[row + block_row][column + block_column]

                if (block_total != total):
                    return False

        return True

    # Return set of valid numbers from values that do not appear in used
    def filter_values(self, values, used):
        return [number for number in values if number not in used]

    # Return first empty spot on grid (marked with 0)
    def get_spot(self, board, state):
        for row in range(board):
            for column in range(board):
                if state[row][column] == 0:
                    return row, column

    # Filter valid values based on row
    def filter_row(self, state, row):
        number_set = range(1, self.type+1) # Defines set of valid numbers that can be placed on board
        in_row = [number for number in state[row] if (number != 0)]
        options = self.filter_values(number_set, in_row)
        return options

    # Filter valid values based on column
    def filter_col(self, options, state, column):
        in_column = [] # List of valid values in spot's column
        for column_index in range(self.type):
            if state[column_index][column] != 0:
                in_column.append(state[column_index][column])
        options = self.filter_values(options, in_column)
        return options

    # Filter valid values based on quadrants
    def filter_quad(self, options, state, row, column):
        in_block = [] # List of valid values in spot's quadrant
        row_start = int(row/self.height)*self.height
        column_start = int(column/3)*3
        
        for block_row in range(0, self.height):
            for block_column in range(0,3):
                in_block.append(state[row_start + block_row][column_start + block_column])
        options = self.filter_values(options, in_block)
        return options    

    def actions(self, state, row, column):
        options = self.filter_row(state, row)
        options = self.filter_col(options, state, column)
        options = self.filter_quad(options, state, row, column)

        return options # Return spot's viable options

def backtracking(board):
    problem = Problem(board)
    if problem.goal_test(board):
        return board

    # Get first empty spot
    row,column = problem.get_spot(problem.type, board)
    # Get spot's viable options
    options = problem.actions(board, row, column)

    for option in options:
        board[row][column] = option # Try viable option
        # Recursively fill in the board
        if backtracking(board):
            return board # Returns board if success 
        else:
            board[row][column] = 0 # Otherwise backtracks

    return None

def solve_bdfs(board):

    print ("\nSolving with Backtracking DFS...")
    start_time = time.time()
    solution = backtracking(board)
    elapsed_time = time.time() - start_time
    
    switcher = {
        0: "          ",
        1: "      \   ",
        2: "       \  ",
        3: "--------\ ",
        4: "         >",
        5: "--------/ ",
        6: "       /  ",
        7: "      /   ",
        8: "          ",
    }

    if solution:
        print ("Found solution")
        for i, row in enumerate(solution):
            print (str(board[i]) + switcher.get(i) + str(row))
    else:
        print ("No possible solutions")

    print ("Elapsed time: " + str(elapsed_time))

#Testing on invalid 9x9 board...
board = [[0,0,9,0,7,0,0,0,5],
      [0,0,2,1,0,0,9,0,0],
      [1,0,0,0,2,8,0,0,0],
      [0,7,0,0,0,5,0,0,1],
      [0,0,8,5,1,0,0,0,0],
      [0,5,0,0,0,0,3,0,0],
      [0,0,0,0,0,3,0,0,6],
      [8,0,0,0,0,0,0,0,0],
      [2,1,0,0,0,0,0,8,7]]

print ("Problem:")
for row in board:
      print (row)

solve_bdfs(board)

#Testing on valid 9x9 board...
board = [[0,0,0,8,4,0,6,5,0],
      [0,8,0,0,0,0,0,0,9],
      [0,0,0,0,0,5,2,0,1],
      [0,3,4,0,7,0,5,0,6],
      [0,6,0,2,5,1,0,3,0],
      [5,0,9,0,6,0,7,2,0],
      [1,0,8,5,0,0,0,0,0],
      [6,0,0,0,0,0,0,4,0],
      [0,5,2,0,8,6,0,0,0]]

print ("Problem:")
for row in board:
      print (row)

solve_bdfs(board)

#Testing on filled valid 9x9 board...
board = [[9,7,4,2,3,6,1,5,8],
      [6,3,8,5,9,1,7,4,2],
      [1,2,5,4,8,7,9,3,6],
      [3,1,6,7,5,4,2,8,9],
      [7,4,2,9,1,8,5,6,3],
      [5,8,9,3,6,2,4,1,7],
      [8,6,7,1,2,5,3,9,4],
      [2,5,3,6,4,9,8,7,1],
      [4,9,1,8,7,3,6,2,5]]

print ("Problem:")
for row in board:
      print (row)

solve_bdfs(board)

# Testing on valid 9x9 board...
board = [[3,0,5,4,2,0,8,1,0],
      [4,8,7,9,0,1,5,0,6],
      [0,2,9,0,5,6,3,7,4],
      [8,5,0,7,9,3,0,4,1],
      [6,1,3,2,0,8,9,5,7],
      [0,7,4,0,6,5,2,8,0],
      [2,4,1,3,0,9,0,6,5],
      [5,0,8,6,7,0,1,9,2],
      [0,9,6,5,1,2,4,0,8]]

print ("Problem:")
for row in board:
      print (row)

solve_bdfs(board)

# Testing unsolvable quadrant on a 9x9 board...
board = [[0,9,0,3,0,0,0,0,1],
      [0,0,0,0,8,0,0,4,6],
      [0,0,0,0,0,0,8,0,0],
      [4,0,5,0,6,0,0,3,0],
      [0,0,3,2,7,5,6,0,0],
      [0,6,0,0,1,0,9,0,4],
      [0,0,1,0,0,0,0,0,0],
      [5,8,0,0,2,0,0,0,0],
      [2,0,0,0,0,7,0,6,0]]

print ("Problem:")
for row in board:
      print (row)

solve_bdfs(board)

