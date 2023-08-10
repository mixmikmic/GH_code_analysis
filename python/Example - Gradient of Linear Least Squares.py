get_ipython().run_line_magic('matplotlib', 'inline')

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

from sympy import MatrixSymbol, Matrix

sp.init_printing(order='rev-lex',use_latex='mathjax')

A = MatrixSymbol('A', 2, 2)
x = MatrixSymbol('x', 2, 1)
b = MatrixSymbol('b', 2, 1)

Ax_minus_b_vector = (A*x - b)
Ax_minus_b_vector.as_explicit()

def l2_norm_on_R2(v_component_0, v_component_1):
    return sp.sqrt(sp.Add(sp.Pow(v_component_0,2),sp.Pow(v_component_1,2)))

def create_f_x(v_component_0,v_component_1):
    return sp.Mul(sp.Rational(1, 2), sp.Pow(l2_norm_on_R2(v_component_0,v_component_1), 2))

A_00,A_01,A_10,A_11 = sp.symbols('A_00 A_01 A_10 A_11')
x_0, x_1 = sp.symbols('x_0 x_1')
b_0, b_1 = sp.symbols('b_0 b_1')

Ax_minus_b_0 = A_00 * x_0 + A_01 * x_1 - b_0
Ax_minus_b_1 = A_10 * x_0 + A_11 * x_1 - b_1

l2_norm_on_R2(Ax_minus_b_0, Ax_minus_b_1)

f_x = create_f_x(Ax_minus_b_0, Ax_minus_b_1)
f_x

partial_f_wrt_x0 = sp.diff(f_x, x_0)
partial_f_wrt_x0

partial_f_wrt_x1 = sp.diff(f_x, x_1)
partial_f_wrt_x1

Matrix([partial_f_wrt_x0, partial_f_wrt_x1])

(A.T * (A*x - b)).as_explicit()

