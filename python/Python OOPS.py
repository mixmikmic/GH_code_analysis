class A:
    __slots__ = ['a','b']
    def __init__(self):
        self.a = 10
        self.b = 20
        pass
i = A()
i.a
i.c = 30

bbs = '01110011001000000110111001101111001000000010000001101001001000000111001101101110001000000110010100100000001000000110100000100000001000000110010100100000011100100010000000100000011100000110110100100000011011110010000001100011'
octets = list(map(lambda i: bbs[i:8+i],range(0,len(bbs),8)))

''.join(list(map(lambda x:chr(int(x,2)),octets)))

quad = list(map(lambda i:bbs[i:i+4],range(0,len(bbs),4)))
''.join(list(map(lambda x:chr(int(x,2)),quad)))



