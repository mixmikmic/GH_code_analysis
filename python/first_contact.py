get_ipython().run_line_magic('display', 'latex')

e^(i*pi) + 1

n(pi, digits=1000)

a = exp(pi*sqrt(163))
a

n(a, digits=50)

n(a, digits=50).str(no_sci=2)

f = diff(sin(x^2),x) ; f

print(latex(f))

integrate(x^5/(x^3-2*x+1), x, hold=True) 

integrate(x^5/(x^3-2*x+1), x) 

f = _
f

diff(f, x)

_.simplify_full()

integrate(exp(-x^2), x, -oo, +oo)

get_ipython().run_line_magic('pinfo', 'diff')

get_ipython().run_line_magic('pinfo2', 'diff')

exp(x).series(x==0, 8)

var('n') # declaring n as a symbolic variable
sum(1/n^2, n, 1, +oo)

sum(1/n^3, n, 1, +oo)

numerical_approx(_)

for n in range(10): 
    print([binomial(n,p) for p in range(n+1)])

n = 2^61-1; n

n.is_prime()

