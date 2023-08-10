get_ipython().magic('pylab inline')

L=1.0
N=500  # make sure N is even for simpson's rule
A=1.0

def fLeft(x):
    return 2*A*x/L
    
def fRight(x):
    return 2*A*(L-x)/L

def fa_vec(x):
    """
    vector version
    'where(cond, A, B)', returns A when cond is true and B when cond is false.
    """
    return where(x<L/2, fLeft(x), fRight(x))

x=linspace(0,L,N)  # define the 'x' array
h=x[1]-x[0]        # get x spacing
y=fa_vec(x)
title("Vertical Displacement of a Plucked String at t=0")
xlabel("x")
ylabel("y")
plot(x,y)

def basis(x, n):
    return sqrt(2/L)*sin(n*pi*x/L)

for n in range(1,5):
    plot(x,basis(x,n),label="n=%d"%n)
    
legend(loc=3)

def simpson_array(f, h):
    """

Use Simpson's Rule to estimate an integral of an array of
    function samples
    
    f: function samples (already in an array format)
    h: spacing in "x" between sample points
    
    The array is assumed to have an even number of elements.
    
    """
    if len(f)%2 != 0:
        raise ValueError("Sorry, f must be an array with an even number of elements.")
        
    evens =  f[2:-2:2]
    odds = f[1:-1:2]
    return (f[0] + f[-1] + 2*odds.sum() + 4*evens.sum())*h/3.0

def braket(n):
    """
    Evaluate <n|f>
    """
    return simpson_array(basis(x,n)*fa_vec(x),h)

M=20
coefs = [0]
coefs_th = [0]
ys = [[]]
sup = zeros(N)
for n in range(1,M):
    coefs.append(braket(n))   # do numerical integral

    if n%2==0:
        coefs_th.append(0.0)
    else:
        coefs_th.append(4*A*sqrt(2*L)*(-1)**((n-1)/2.0)/(pi**2*n**2))  # compare theory
        
    ys.append(coefs[n]*basis(x,n))
    sup += ys[n]
    plot(x,sup)

print("%10s\t%10s\t%10s" % ('n', 'coef','coef(theory)'))
print("%10s\t%10s\t%10s" % ('---','-----','------------'))

for n in range(1,M):
    print("%10d\t%10.5f\t%10.5f" % (n, coefs[n],coefs_th[n]))




