def print_board(board):
    print(board[0],"|",board[1],"|",board[2])
    print("----------")
    print(board[3],"|",board[4],"|",board[5])
    print("----------")
    print(board[6],"|",board[7],"|",board[8])

def check_win(board):
    
    if board[0]==board[1]==board[2]:
        winner=board[0]
        return(winner,True)
    elif board[0]==board[3]==board[6]:
        winner=board[0]
        return(winner,True)
    elif board[0]==board[4]==board[8]:
        winner=board[0]
        return(winner,True)
    elif board[3]==board[4]==board[5]:
        winner=board[3]
        return(winner,True)
    elif board[6]==board[7]==board[8]:
        winner=board[6]
        return(winner,True)
    elif board[2]==board[5]==board[8]:
        winner=board[2]
        return(winner,True)
    else:
        return(0,False)

import numpy as np

board = np.arange(1,10)

board = list(map(str,board))
    
count=1

print_board(board)


    
while(count<=5):    
    
    print()
    print("Turn {}".format(count))

    print()
    print("Player 1: Please enter location of 'X' (1-9)")
    print()
    value_1 = int(input())
    board[value_1-1] = 'X'
    
    print()
    print("##############################################")
    print()
    print_board(board)
    
    
    if check_win(board)[1] is True:
        print()
        print("Game Over. '{}' wins".format(check_win(board)[0]))
        break
        
    if count == 5 and check_win(board)[1] is not True:
        print()
        print("Game Tied.")
        break
        
    

    print()
    print("Player 2: Please enter location of 'O' (1-9)")
    print()
    value_2 = int(input())
    board[value_2-1] = 'O'

    print()
    print("###############################################")
    print()
    print_board(board)
    

    if check_win(board)[1] is True:
        print()
        print("Game Over. '{}' wins".format(check_win(board)[0]))
        break

    count+=1
    
    
print()
# print("Game tied")



