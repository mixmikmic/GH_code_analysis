def bitperm(n):
    A = [0]*n
    def inner(n,idx):
        nonlocal A
        if n==idx:
            print(''.join(A))
        else:
            A[idx]='0'
            inner(n,idx+1)
            A[idx]='1'
            inner(n,idx+1)
                
                
    inner(n,0)

bitperm(2)

def subsets(s,idx,acc):
    if idx==len(s):
        print(acc)
    else:
        acc.append(s[idx])
        subsets(s,idx+1,acc)
        acc.pop()
        subsets(s,idx+1,acc)

subsets([1,2,3],0,[])

def Permutations(item):
    def inner(item,start,end):
        if start==end:
            print(''.join([str(x) for x in item]))
        else:
            for i in range(start,end):
                item[i],item[start]= item[start],item[i]
                inner(item,start+1,end)
                item[i],item[start]=item[start],item[i]
    inner(item,0,len(item))

Permutations([1,2,3])

from itertools import permutations
for v in permutations('123'):
    print(''.join(v))

N = 4
board = [[0]*N for i in range(N)]

def print_board(board):
    for x in board:
        
        print('|'.join(['.' if i==0 else 'Q' for i in x]))
        print()
    print('----------------')

print_board(board)

def is_safe(board,row,col):
    for i in range(col):
        if board[row][i]:
            return False
        
    i = row
    j = col
    while i >=0 and j >=0:
        if board[i][j]:
            return False
        i -= 1
        j -= 1
    i = row
    j = col
    while j >= 0 and i < len(board):
        if board[i][j]:
            return False
        i += 1
        j -= 1
    return True

def SolveNQueens(N):
    board = [[0]*N for i in range(N)]
    def solven(board,col,N):
        if col == N:
            print_board(board)
        else:
            for i in range(N):
                if is_safe(board,i,col):
                    board[i][col]=1
                    solven(board,col+1,N)
                    board[i][col]=0
    solven(board,0,N)

SolveNQueens(4)

from itertools import permutations
N=4
cols = range(N)

def print_b(vec):
    for col in vec:
        s = ['.']*len(vec)
        s[col]='q'
        print('|'.join(s))
        print(''.join(['__']*len(vec)))
    print('---------------')

for vec in permutations(cols):
    if N == len(set(vec[i]+i for i in cols))          == len(set(vec[i]-i for i in cols)):
        print_b(vec)

def infN():
    x = 1
    while True:
        yield x
        x += 1

g = infN

def MakeEqn(A):
     sym = [None]*(len(A))
     def inner(A,idx,ER,sum):
         ts = sum
         #test A-1 items only because A[-1] is ER
         if idx == len(A):
             if ER==sum:
                i = 0
                print('0 ',end='')
                while i < len(A):
                    print(sym[i],end=' ')
                    print(A[i],end=' ')
                    i+=1
                print(' = ',ER)

         else:
             sum = sum + A[idx]
             sym[idx]='+'
             inner(A,idx+1,ER,sum)
             sym[idx]='-'
             ts = ts - A[idx]
             inner(A,idx+1,ER,ts)
             #sum = sum - A[idx]
             #inner(A,idx+1,ER,sum)
     B = A[0:-1]
     ER = A[-1]
     inner(B,0,ER,0)

for x in range(1,16):
    MakeEqn([1,2,3,4,5,x])

def permu(lst):
    res = []
    def perms(l,s,e):
        nonlocal res
        if s==e:
            res += [list(l)]
        else:
            for x in range(s,e):
                l[s],l[x]=l[x],l[s]
                perms(l,s+1,e)
                l[s],l[x]=l[x],l[s]
    perms(lst,0,len(lst))
    return res

p = permu([1,2,3,4])

p



