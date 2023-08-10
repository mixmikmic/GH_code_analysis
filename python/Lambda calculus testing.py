lamb.reload_all()

x = lang.te("x_e")
y = lang.te("y_e")
z = lang.te("z_e")

test2 = lang.te("L y_e : L x_e : y_e")
test2(x)

test2.apply(x)

meta.alpha_convert_new(test2, {"y"})

test3 = lang.te("L y_e : L y1_e : P(y_e) & q_t")
test3

meta.alpha_convert_new(test3, {"x", "y", "q"})

meta.pmw_test1

meta.pmw_test1b

meta.pmw_test1(meta.t2)

meta.pmw_test1(meta.t2).reduce().derivation

test4 = lang.te("L x_e : x")
test4

test4(x)

r = meta.pmw_test1(meta.t2).reduce()
r.derivation

get_ipython().run_cell_magic('lamb', '', 'collision = lambda x_e : (lambda f_<e,t> : lambda x_e : f(x))(lambda y_e : y <=> x)')

collision.reduce_all().derivation

collision.reduce_all().derivation.trace()

reload_lamb()
meta.TypedExpr.try_parse_op_expr("lambda x_e:P_<e,t>(x)")

lang.te("L y_e : Exists x_e : P(x_e)(y)")(lang.te("x_e")).reduce().derivation

get_ipython().magic('lamb test = (lambda w_n  : Exists q_<n,t> : q(w))')

get_ipython().magic('lamb test = (Lambda w_n  : w)')

get_ipython().run_cell_magic('lamb', '', 'id = ((lambda f_<<e,e>,<e,e>> : lambda g_<e,e> : lambda x_e : (f(g))(x))(lambda h_<e,e> : h))(lambda i_e : i)')

id.reduce_all()

id.reduce_all().derivation

id.reduce_all().derivation.trace()



