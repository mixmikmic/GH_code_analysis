import sympy
sympy.init_printing()

(R, V1, V2, V3, V4, V5, V6, V7, V8, C,
 G1, G2, G3, H1, H2, H3) = sympy.symbols('R, V1, V2, V3, V4, V5, V6, V7, V8, C,'
                                         'G1, G2, G3, H1, H2, H3')
unknowns = V1, V2, V3, V4, V5, V6, V7, V8, C

eqs = [# Blocks
       V2 - G1*V1,
       V4 - G2*V3,
       C - G3*V5,
       V6 - H1*V4,
       V7 - H2*V4,
       V8 - H3*C,
       # Sums
       V1 - (R - V6),
       V3 - (V2 - V7),
       V5 - (V4 + V3 - V8),
       ]

sol = sympy.solve(eqs, unknowns)
sol

sol[C].factor()



