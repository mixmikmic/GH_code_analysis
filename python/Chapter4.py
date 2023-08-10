get_ipython().magic('matplotlib inline')

#P96/97: Basic factorization and expansion
from sympy import Symbol, factor, expand
x = Symbol('x')
y = Symbol('y')
expr = x**2 - y**2
f = factor(expr)
print(f)
# Expand
print(expand(f))

#P97: Factorizing and expanding a complicated identity
from sympy import Symbol, factor, expand
x = Symbol('x')
y = Symbol('y')
expr = x**3 + 3*x**2*y + 3*x*y**2 + y**3 

print('Original expression: {0}'.format(expr))
factors = factor(expr)
print('Factors: {0}'.format(factors))


expanded = expand(factors)
print('Expansion: {0}'.format(expanded))

#P97: Pretty printing
from sympy import Symbol, pprint, init_printing
x = Symbol('x')
expr = x*x + 2*x*y + y*y
pprint(expr)
# Reverse order lexicographical
init_printing(order='rev-lex')
expr = 1 + 2*x + 2*x**2
pprint(expr)

#P99: Print a series

'''
Print the series:
x + x**2 + x**3 + ... + x**n
    ____  _____         ____    
      2     3             n
'''
from sympy import Symbol, pprint, init_printing
def print_series(n):
    # initialize printing system with
    # reverse order
    init_printing(order='rev-lex')
    x = Symbol('x')
    series = x
    for i in range(2, n+1):
        series = series + (x**i)/i
    pprint(series)

if __name__ == '__main__':
    n = input('Enter the number of terms you want in the series: ')
    print_series(int(n))

#P100: Substituting in values
from sympy import Symbol
x = Symbol('x')
y = Symbol('y')
expr = x*x + x*y + x*y + y*y
res = expr.subs({x:1, y:2})
res

#P102: Print a series and also calculate its value at a certain point

'''
Print the series:

x + x**2 + x**3 + ... + x**n
    ____  _____         ____    
      2     3             n
      
and calculate its value at a certain value of x.
'''

from sympy import Symbol, pprint, init_printing
def print_series(n, x_value):
    # initialize printing system with
    # reverse order
    init_printing(order='rev-lex')
    x = Symbol('x')
    series = x
    for i in range(2, n+1):
        series = series + (x**i)/i
    pprint(series)
    # evaluate the series at x_value
    series_value = series.subs({x:x_value})
    print('Value of the series at {0}: {1}'.format(x_value, series_value))

if __name__ == '__main__':
    n = input('Enter the number of terms you want in the series: ')
    x_value = input('Enter the value of x at which you want to evaluate the series: ') 
    print_series(int(n), float(x_value))

# P104: Expression multiplier

'''
Product of two expressions
'''

from sympy import expand, sympify
from sympy.core.sympify import SympifyError
def product(expr1, expr2):
    prod = expand(expr1*expr2)
    print(prod)

if __name__=='__main__':
    expr1 = input('Enter the first expression: ')
    expr2 = input('Enter the second expression: ')
    try:
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
    except SympifyError:
        print('Invalid input')
    else:
        product(expr1, expr2)

#P105: Solving a linear equation
from sympy import Symbol, solve 
x = Symbol('x')
expr = x - 5 - 7
solve(expr)

#P106: Solving a quadratic equation
from sympy import solve 
x = Symbol('x')
expr = x**2 + 5*x + 4 
solve(expr, dict=True)

#P106: Quadratic equation with imaginary roots
from sympy import Symbol
x=Symbol('x')
expr = x**2 + x + 1
solve(expr, dict=True)

#P106/107: Solving for one variable in terms of others
from sympy import Symbol, solve
x = Symbol('x')
a = Symbol('a') 
b = Symbol('b')
c = Symbol('c')
expr = a*x*x + b*x + c
solve(expr, x, dict=True)

#P107: Express s in terms of u, a, t
from sympy import Symbol, solve, pprint 
s = Symbol('s')
u = Symbol('u')
t = Symbol('t')
a = Symbol('a')
expr = u*t + (1/2)*a*t*t - s
t_expr = solve(expr,t, dict=True) 
t_expr

#P108: Solve a system of Linear equations
from sympy import Symbol
x = Symbol('x')
y = Symbol('y')
expr1 = 2*x + 3*y - 6 
expr2 = 3*x + 2*y - 12
solve((expr1, expr2), dict=True)

#P109: Simple plot with SymPy
from sympy.plotting import plot
from sympy import Symbol
x = Symbol('x')
plot(2*x+3)

#P110: Plot in SymPy with range of x as well as other attributes specified
from sympy import plot, Symbol
x = Symbol('x')
plot(2*x + 3, (x, -5, 5), title='A Line', xlabel='x', ylabel='2x+3')

#P112: Plot the graph of an input expression
'''
Plot the graph of an input expression
'''
from sympy import Symbol, sympify, solve
from sympy.plotting import plot

def plot_expression(expr):
    y = Symbol('y')
    solutions = solve(expr, y)
    expr_y = solutions[0]
    plot(expr_y)

if __name__=='__main__':
    expr = input('Enter your expression in terms of x and y: ')
    try:
        expr = sympify(expr)
    except SympifyError:
        print('Invalid input')
    else:
        plot_expression(expr)

#P113: Plotting multiple functions
from sympy.plotting import plot 
from sympy import Symbol
x = Symbol('x')
plot(2*x+3, 3*x+1)

#P114: Plot of the two lines drawn in a different color
from sympy.plotting import plot 
from sympy import Symbol
x = Symbol('x')
p = plot(2*x+3, 3*x+1, legend=True, show=False) 
p[0].line_color = 'b'
p[1].line_color = 'r'
p.show()

#P116: Example of summing a series
from sympy import Symbol, summation, pprint 
x = Symbol('x')
n = Symbol('n')
s = summation(x**n/n, (n, 1, 5)) 
s.subs({x:1.2})

#P117: Example of solving a polynomial inequality
from sympy import Poly, Symbol, solve_poly_inequality
x = Symbol('x')
ineq_obj = -x**2 + 4 < 0 
lhs = ineq_obj.lhs
p = Poly(lhs, x)
rel = ineq_obj.rel_op
solve_poly_inequality(p, rel)

#P118: Example of solving a rational inequality
from sympy import Symbol, Poly, solve_rational_inequalities
x = Symbol('x')
ineq_obj = ((x-1)/(x+2)) > 0
lhs = ineq_obj.lhs
numer, denom = lhs.as_numer_denom()
p1 = Poly(numer)
p2 = Poly(denom)
rel = ineq_obj.rel_op
solve_rational_inequalities([[((p1, p2), rel)]])

#P118: Solve a non-polynomial inequality
from sympy import Symbol, solve, solve_univariate_inequality, sin 
x = Symbol('x')
ineq_obj = sin(x) - 0.6 > 0
solve_univariate_inequality(ineq_obj, x, relational=False)

