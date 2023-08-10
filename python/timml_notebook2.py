from timml import *
from pylab import *
get_ipython().magic('matplotlib notebook')

# Create basic model elements
ml = Model(k=[2, 6, 4],
           zb=[140, 80, 0],
           zt=[165, 120, 60],
           c=[2000, 20000],
           n=[0.3, 0.25, 0.3],
           nll=[0.2, 0.25])
rf = Constant(ml, xr=20000, yr=20000, head=175, layer=0)
p = CircAreaSink(ml, xp=10000, yp=10000, Rp=15000, infil=0.0002, layer=0)
w1 = Well(ml, xw=10000, yw=8000, Qw=1000, rw=0.3, layers=0, label='well 1')
w2 = Well(ml, xw=12100, yw=10700, Qw=5000, rw=0.3, layers=2, label='well 2')
w3 = Well(ml, xw=10000, yw=4600, Qw=5000, rw=0.3, layers=[1,2], label='maq well')
#
HeadLineSink(ml, x1=9510, y1=19466, x2=12620, y2=17376, head=170, layers=0)
HeadLineSink(ml, 12620, 17376, 12753, 14976, 168, [0])
HeadLineSink(ml, 12753, 14976, 13020, 12176, 166, [0])
HeadLineSink(ml, 13020, 12176, 15066, 9466,  164, [0])
HeadLineSink(ml, 15066, 9466,  16443, 7910,  162, [0])
HeadLineSink(ml, 16443, 7910,  17510, 5286,  160, [0])
HeadLineSink(ml, 17510, 5286,  17600, 976,   158, [0])
#
HeadLineSink(ml, 356,   6976,  4043,  7153, 174, [0])
HeadLineSink(ml, 4043,  7153,  6176,  8400, 171, [0])
HeadLineSink(ml, 6176,  8400,  9286,  9820, 168, [0])
HeadLineSink(ml, 9286,  9820,  12266, 9686, 166, [0])
HeadLineSink(ml, 12266, 9686,  15066, 9466, 164, [0])
#
HeadLineSink(ml, 1376,  1910,  4176,  2043, 170, [0])
HeadLineSink(ml, 4176,  2043,  6800,  1553, 166, [0])
HeadLineSink(ml, 6800,  1553,  9953,  2086, 162, [0])
HeadLineSink(ml, 9953,  2086,  14043, 2043, 160, [0])
HeadLineSink(ml, 14043, 2043,  17600, 976 , 158, [0])
#
ResLineSink(ml, x1=12753, y1=14976, x2=10781, y2=14895, head=168, res=5, width=10, layers=[0])
ResLineSink(ml, 10781, 14895, 8385,  15677, 170, 5, 10, [0])
ResLineSink(ml,  8385, 15677, 6094,  15885, 172, 5, 10, [0])
ResLineSink(ml,  6094, 15885, 3229,  14843, 174, 5, 10, [0])
ls = ResLineSink(ml,  3229, 14843,  833,  14261, 176, 5, 10, [0])
ml.solve()

timlayout(ml, width=2)  # to make wider lines
timcontour(ml, 0, 20000, 100, 0, 20000, 100, 3, levels=10, newfig=False, width=1, size=(6,6))

print 'the head at the center of ResLineSink ls is:', ml.head(0, ls.xcp, ls.ycp)
print 'the head in the river is specified as:', 176

print 'The head at well 1 is:', ml.headVector(w1.xw, w1.yw)
print 'The head at well 2 is:', ml.headVector(w2.xw, w2.yw)
print 'The head at well 3 is:', ml.headVector(w3.xw, w3.yw)
print 'The discharge of well 1 is:', w1.parameters
print 'The discharge of well 2 is:', w2.parameters
print 'The discharge of well 3 is:', w3.parameters

# Create a contour plot for the area
timcontour(ml, 0, 20000, 50, 0, 20000, 50, 3, levels=10, xsec=True, size=(8,8))

# Capture zone is not working in Ipython Notebook yet!
capturezone(ml, w=w1, N=10, z=150, tmax=50*365.25, xsec=True)
capturezone(ml, w=w2, N=10, z=30, tmax=50*365.25, xsec=True)
capturezone(ml, w=w3, N=10, z=30, tmax=50*365.25, xsec=True)
capturezone(ml, w=w3, N=10, z=100, tmax=50*365.25, xsec=True)

