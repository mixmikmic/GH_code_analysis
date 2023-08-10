a = 3.0
c = 1.0
x1, x2, x3 = 2.0, 2.0, 2.0
print "%6d %18.8e %18.8e %18.8e" % (0,x1,x2,x3)
for i in range(5):
    x1 = x1 + c*(x1**2 - a)
    x2 = a/x2
    x3 = 0.5*(x3 + a/x3)
    print "%6d %18.8e %18.8e %18.8e" % (i+1,x1,x2,x3)

