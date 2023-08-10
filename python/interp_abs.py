get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import linspace,polyfit,polyval,cos,pi
from matplotlib.pyplot import figure,plot,subplot,legend,axis,text

def interp(points):
    xmin, xmax = -1.0, +1.0
    xx = linspace(xmin,xmax,100);
    ye = abs(xx);

    figure(figsize=(10,8))
    for i in range(1,7):
        N = 2*i
        subplot(3,2,i)
        if points == 'uniform':
            x = linspace(xmin,xmax,N+1)
        else:
            theta = linspace(0,pi,N+1)
            x = cos(theta)
        y = abs(x);
        P = polyfit(x,y,N);
        yy = polyval(P,xx);
        plot(x,y,'o',xx,ye,'--',xx,yy)
        axis([xmin, xmax, -0.1, +1.1])
        text(-0.1,0.5,'N = '+str(N))

interp('uniform')

interp('chebyshev')

