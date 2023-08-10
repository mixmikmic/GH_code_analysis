get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import pi,cos,linspace,polyfit,polyval
from matplotlib.pyplot import figure,subplot,plot,axis,text

xmin, xmax = 0.0, 2.0*pi
fun = lambda x: cos(x)

xx = linspace(xmin,xmax,100);
ye = fun(xx);

figure(figsize=(9,8))
for i in range(1,7):
    N = 2*i;
    subplot(3,2,i)
    x = linspace(xmin,xmax,N+1);
    y = fun(x);
    P = polyfit(x,y,N);
    yy = polyval(P,xx);
    plot(x,y,'o',xx,ye,'--',xx,yy);
    axis([xmin, xmax, -1.1, +1.1])
    text(3.0,0.5,'N='+str(N))

