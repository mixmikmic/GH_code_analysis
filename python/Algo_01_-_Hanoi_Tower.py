def hanoi(height, fro='left', to='right', through='middle'):
    if height:
        hanoi(height - 1, fro, through, to )
        print ('%-7s => %s' % (fro, to))
        hanoi(height -1, through, to, fro)

hanoi(2)

hanoi(3)

hanoi(4)

hanoi(5)



