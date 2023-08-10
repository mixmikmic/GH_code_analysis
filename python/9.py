def triplet(equal):
    for i in range(1,1000):
        for j in range(1,1000):
            c = 1000 - i - j 
            if i**2 + j**2 == c**2:
                return i, j, c, i*j*c

triplet(1000)

