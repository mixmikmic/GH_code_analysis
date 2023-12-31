w = [0.0, 1.1, 2.2, 0.0, 2.1, 2.2, 0.2]
l = {(0,1): 1.0, (1,2): 1.0, (2,3): 1.0, (1,4): 1.0, (4,5): 0.3, (5,2): 1.0, (5,6): 0.5, (1,3): 8.0}
f = {0: (0.0,1.0), 3: (2.0,1.0)}
g = 9.81

from mosek.fusion import *

# w - masses of points
# l - lengths of strings
# f - coordinates of fixed points
# g - gravitational constant
def stringModel(w, l, f, g):
    n, m = len(w), len(l)
    starts = [ lKey[0] for lKey in l.keys() ]
    ends = [ lKey[1] for lKey in l.keys() ]

    M = Model("strings")

    # Coordinates of points
    x = M.variable("x", [n, 2])

    # A is the signed incidence matrix of points and strings
    A = Matrix.sparse(m, n, range(m)+range(m), starts+ends, [1.0]*m+[-1.0]*m)

    # ||x_i-x_j|| <= l_{i,j}
    c = M.constraint("c", Expr.hstack(Expr.constTerm(l.values()), Expr.mul(A, x)), 
        Domain.inQCone() )

    # x_i = f_i for fixed points
    for i in f:
        M.constraint(x.slice([i,0], [i+1,2]), Domain.equalsTo(list(f[i])))

    # sum (g w_i x_i_2)
    M.objective(ObjectiveSense.Minimize, 
        Expr.mul(g, Expr.dot(w, x.slice([0,1], [n,2]))))

    # Solve
    M.solve()
    if M.getProblemStatus(SolutionType.Interior) == ProblemStatus.PrimalAndDualFeasible:
        return x.level().reshape([n,2]), c.dual().reshape([m,3])
    else:
        return None, None

get_ipython().magic('matplotlib inline')
# x - coordinates of the points
# c - dual values of string length constraints
# d - pairs of points to connect
def display(x, c, d):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # Plot points
    ax.scatter(x[:,0], x[:,1], color="r")
    # Plot fully stretched strings (nonzero dual value) as solid lines, else dotted lines
    for i in range(len(c)):
        col = "b" if c[i][0] > 1e-4 else "b--"
        ax.plot([x[d[i][0]][0], x[d[i][1]][0]], [x[d[i][0]][1], x[d[i][1]][1]], col)
    ax.axis("equal")
    plt.show()

x,c = stringModel(w, l, f, g)

if x is not None:
    display(x, c, l.keys())

n = 1000
w = [1.0]*n
l = {(i,i+1): 1.0/n for i in range(n-1)}
f = {0: (0.0,1.0), n-1: (0.7,1.0)}
g = 9.81

x,c = stringModel(w, l, f, g)
if x is not None:
    display(x, c, l.keys())

n = 20
w = [1.0]*n
l = {(i,i+1): 0.09 for i in range(n-1)}
l.update({(5,14): 0.3})
f = {0: (0.0,1.0), 13: (0.5,0.9), 17: (0.7,1.1)}
g = 9.81

x,c = stringModel(w, l, f, g)
if x is not None:
    display(x, c, l.keys())

def dualStringModel(w, l, f, g):
    n, m = len(w), len(l)
    starts = [ lKey[0] for lKey in l.keys() ]
    ends = [ lKey[1] for lKey in l.keys() ]

    M = Model("dual strings")

    x = M.variable(Domain.inQCone(m,3))       #(y,v)
    y = x.slice([0,0],[m,1])
    v = x.slice([0,1],[m,3])
    z = M.variable([n,2])

    # z_i = 0 if i is not fixed
    for i in range(n):
        if i not in f:
            M.constraint(z.slice([i,0], [i+1,2]), Domain.equalsTo(0.0))

    B = Matrix.sparse(m, n, range(m)+range(m), starts+ends, [1.0]*m+[-1.0]*m).transpose()
    w2 = Matrix.sparse(n, 2, range(n), [1]*n, [-wT*g for wT in w])

    # sum(v_ij *sgn(ij)) + z_i = -(0, gw_i) for all vertices i
    M.constraint(Expr.add( Expr.mul(B, v), z ), Domain.equalsTo(w2))

    # Objective -l*y -fM*z
    fM = Matrix.sparse(n, 2, f.keys()+f.keys(), [0]*len(f)+[1]*len(f), 
                       [pt[0] for pt in f.values()] + [pt[1] for pt in f.values()])
    
    M.objective(ObjectiveSense.Maximize, Expr.neg(Expr.add(Expr.dot(l.values(), y),Expr.dot(fM, z))))
    M.solve()

# w - masses of points
# l - lengths of strings
# f - coordinates of fixed points
# g - gravitational constant
# k - stiffness coefficient
def elasticModel(w, l, f, g, k):
    n, m = len(w), len(l)
    starts = [ lKey[0] for lKey in l.keys() ]
    ends = [ lKey[1] for lKey in l.keys() ]

    M = Model("strings")
    x = M.variable("x", [n, 2])                 # Coordinates
    t = M.variable(m, Domain.greaterThan(0.0))  # Streching

    T = M.variable(1)                           # Upper bound
    M.constraint(Expr.vstack(T, Expr.constTerm(1.0), t), Domain.inRotatedQCone())

    # A is the signed incidence matrix of points and strings
    A = Matrix.sparse(m, n, range(m)+range(m), starts+ends, [1.0]*m+[-1.0]*m)

    # ||x_i-x_j|| <= l_{i,j} + t_{i,j}
    c = M.constraint("c", Expr.hstack(Expr.add(t, Expr.constTerm(l.values())), Expr.mul(A, x)), 
        Domain.inQCone() )

    # x_i = f_i for fixed points
    for i in f:
        M.constraint(x.slice([i,0], [i+1,2]), Domain.equalsTo(list(f[i])))

    # sum (g w_i x_i_2) + k*T
    M.objective(ObjectiveSense.Minimize, 
        Expr.add(Expr.mul(k,T), Expr.mul(g, Expr.dot(w, x.slice([0,1], [n,2])))))

    # Solve
    M.solve()
    if M.getProblemStatus(SolutionType.Interior) == ProblemStatus.PrimalAndDualFeasible:
        return x.level().reshape([n,2]), c.dual().reshape([m,3])
    else:
        return None, None

n = 20
w = [1.0]*n
l = {(i,i+1): 0.09 for i in range(n-1)}
l.update({(5,14): 0.3})
f = {0: (0.0,1.0), 13: (0.5,0.9), 17: (0.7,1.1)}
g = 9.81
k = 800

x, c = elasticModel(w, l, f, g, k)
if x is not None:
    display(x, c, l.keys())

