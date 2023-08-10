import sympy
sympy.init_printing()

G_p11, G_p12, G_p21, G_p22, G_c1, G_c2 = sympy.symbols('G_p11, G_p12, G_p21, G_p22, G_c1, G_c2')

G_p = sympy.Matrix([[G_p11, G_p12],
                    [G_p21, G_p22]])

G_c = sympy.Matrix([[G_c1, 0],
                    [0, G_c2]])

I = sympy.Matrix([[1, 0],
                  [0, 1]])

Gamma = sympy.simplify((I + G_p*G_c).inv()*G_p*G_c)

Gamma[0, 0]

Gamma[0, 1]

Delta = (I + G_p*G_c).det()
Delta

(Delta*Gamma).simplify()

G_c*sympy.Matrix([[0, 1], [1, 0]])

