reload_lamb()

unify(tp("[e|t]"), tp("[e|t]"))

tp("[t|e]")

unify(tp("[e|t|<e,t>]"), tp("[e|t|n]"))

unify(tp("[e|t]"), tp("e"))

unify(tp("[e|t]"), tp("[t|n]"))

try:
    tp("[e]")
except types.TypeParseError as e:
    print(e)

try:
    tp("[e|e]")
except types.TypeParseError as e:
    print(e)

tp("[e|e|t]")

tp("[e|[n|t]]")

unify(tp("[e|t]"), tp("n"))

unify(tp("[<e,t>|<e,n>]"), tp("X"))

unify(tp("[<e,t>|<e,n>|<n,t>]"), tp("<X,t>"))

unify(tp("[<e,t>|<e,n>|<n,<e,t>>]"), tp("<X,<Y,t>>"))

unify(tp("<e,[t|n]>"), tp("[<e,t>|<e,n>]"))

tp("[<e,t>|<e,n>]").factor_functional_types() # convert to a functional type.  Will return None if no such conversion is possible.

tp("[<e,t>|<n,t>]").factor_functional_types()

get_ipython().run_cell_magic('lamb', '', '||equals|| = L x_[e|n] : Equivalent(x)\nx = x_e')

equals.content(x) # forces narrowing of the argument type

get_ipython().magic('te P_<[e|n],t>(x_[n|t]) # forces narrowing of both types')

get_ipython().magic("te (L x_[e|n] : P_<[e|n],t>(x))(C_n) # forces narrowing including of both x and, indirectly, the predicate's type")

te("(L x_[e|n] : P_<[e|n|t],t>(x))(C_X)")

x = get_ipython().magic('te Disjunctive(A_e, B_n, C_t)')
x

x.try_adjust_type(tp("e"))

x.try_adjust_type(tp("[e|t]"))

x.try_adjust_type(tp("[e|t]")).try_adjust_type(tp("n")) # leads to None



te("Disjunctive(x_e, y_[<e,t>|n], z_t)").try_adjust_type(tp("[e|t]"))

get_ipython().magic('te Disjunctive(x_e, y_e)')

get_ipython().magic('te Disjunctive(x_e, y_e, z_[e|t])')

r = get_ipython().magic('te Disjunctive(x_e, Disjunctive(y_n, z_t))')
r

r.try_adjust_type(tp("n"))

r.try_adjust_type(tp("[e|n]"))

r.try_adjust_type(tp("[n|t]"))

get_ipython().magic('te Disjunctive(x_e, y_[<e,t>|n])')

f = get_ipython().magic('te L x_e : Disjunctive(x_[e|n], False_t)')
f

f.type

get_ipython().magic('te reduce (L x_e : Disjunctive(x_[e|n], y_t))(A_e)')

get_ipython().magic('te reduce (L x_[e|t] : P(x) & x)((L x_e : Disjunctive(x_[e|n], y_t))(A_e))')

get_ipython().magic('te reduce (L x_e : Disjunctive(x_[e|n], False_t))(A_e) & P_<e,t>((L x_e : Disjunctive(x_[e|n], False_t))(A_e))')



f = get_ipython().magic('te Disjunctive((L f_<e,t> : f), (L f_<e,t> : L g_<e,t> : Exists x_e : f(x) & g(x)))')
f

f(te("P_<e,t>")).reduce_all()

get_ipython().magic('te reduce (Disjunctive((L x_e : x), P_<e,t>))(A_e)')

get_ipython().magic('te reduce (Disjunctive((L x_e : x), P_<n,t>))(A_e)')

get_ipython().magic('te reduce (Disjunctive((L x_e : x), P_n))(A_e)')

get_ipython().magic('te reduce (Disjunctive((L x_n : x), P_n))(A_e)')



get_ipython().magic('te L x_[e|n] : Disjunctive(x_e, x_n)')

get_ipython().magic('te L x_[e|n] : Disjunctive(P_<e,e>(x), Q_<n,t>(x))')



