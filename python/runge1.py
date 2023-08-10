get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import exp,linspace,polyfit,polyval,cos,pi
from matplotlib.pyplot import figure,plot,legend,axis,text,subplot

xmin, xmax = -1.0, +1.0

f1 = lambda x: exp(-5.0*x**2)
f2 = lambda x: 1.0/(1.0+16.0*x**2)

xx = linspace(xmin,xmax,100,True)
figure(figsize=(8,4))
plot(xx,f1(xx),xx,f2(xx))
legend(("$1/(1+16x^2)$", "$\exp(-5x^2)$"));

def interp(f,points):
    xx = linspace(xmin,xmax,100,True);
    ye = f(xx);

    figure(figsize=(9,8))
    for i in range(1,7):
        N = 2*i
        subplot(3,2,i)
        if points == 'uniform':
            x = linspace(xmin,xmax,N+1,True)
        else:
            theta = linspace(0,pi,N+1, True)
            x = cos(theta)
        y = f(x);
        P = polyfit(x,y,N);
        yy = polyval(P,xx);
        plot(x,y,'o',xx,ye,'--',xx,yy)
        axis([xmin, xmax, -1.0, +1.1])
        text(-0.1,0.0,'N = '+str(N))

interp(f1,'uniform')

interp(f2,'uniform')

interp(f2,'chebyshev')

interp(f1,'chebyshev')

