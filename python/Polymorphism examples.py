reload_lamb()

get_ipython().magic('te P_<e,t>(x_e) & P_<n,t>(n_n) # should fail because no principle type for P')

get_ipython().magic('te L x_e : Q_<X,X>(x_X)')

get_ipython().magic('te Q_<X,X>(x_e)')

get_ipython().magic('te p_<e,t>(x_e) & p_<X,t>(n_X) # variables of the same name should have the same type')

get_ipython().magic('te Q_X(x_e)')

get_ipython().magic('te (L x_X : P_<X,X>(x))(x_e)')

get_ipython().magic('te (f_<t,t>(p_<X,X>(Q_<X,X>(x_X))))')

get_ipython().magic('te (f_<<X,Y>,<X,Y>>(g_<X,X>))')

get_ipython().magic('te (L f_<X,Y> : L x_X: y_Y)(g_<Z,Z>)')

get_ipython().magic("te p_X(q_X'(a_X''(b_X'''(c_X''''(d_e)))))")

get_ipython().magic('te L x_X : x(y_<e,t>)')

get_ipython().magic('te (L g_<Y,t> : L x_X : g(x)) (L x_X10 : P_<Z,t>(x)) # application involving two LFuns')

get_ipython().magic('te P_<e,t>(x_X) & Q_<X,Y>(x_X)')

get_ipython().magic('te p_t & Q_<X,X>(x_X)')

x = get_ipython().magic('te Iota x_X: P(x)')
x.try_adjust_type(tp("e"))

x.try_adjust_type(tp("Y"))

get_ipython().magic('te (L x_X : P_<e,t>(x))(Iota x_Y: Q_<Y,t>(x))')

get_ipython().magic('te L x_X : x_e')

get_ipython().magic('te L x_e : x_X')

get_ipython().magic('te L x_X : z_X(y_e)')

get_ipython().magic('te L x_X : x_X & x_X')

get_ipython().magic('te L x_X : g_<Y,t>(x)')

get_ipython().magic('te L g_<Y,t> : L x_X : g(x) # across multiple LFuns')

get_ipython().magic("te L x_X : (L y_Y : (L a_Z : a(z_e))(y))(x) & p_X'")

get_ipython().magic("te L a_X : L b_X' : a(b) # application should force functional type")

get_ipython().magic("te L a_X : L c_X'' : a((L b_X' : b)(c)) # X'' = X', X = <X', ?>")

get_ipython().magic("te L d_X22 : L c_X' : L b_X'' : L a_X''' : L q_X'''' : L p_X''''' : p(q(a(b(c(d)))))")

get_ipython().magic('te L x_X : (L y_<Y,Z> : y(z_e))(x) & p_X # X = <Y,Z> (appl), Y = <e,?> (appl), Z = t (conjunction), X=<e,t> (unification), X = t (conjunction)')

get_ipython().magic('te L x_X : P_<Y,t>(A_Y) & Q_<Y,t>(x_e) # Y=X (appl), X=e (from var), inference to type vars not shared on bound variables')

get_ipython().magic('te (L x_X : P_<Y,t>(A_Y) & Q_<X,t>(x_e))((L x_Y : x)(A_Y))')

# ex of geach rule: complex inference to parts of function
get_ipython().magic('te (λ g_<Y,Z>: (λ f_<X,Y>: (λ x_X: g_<Y,Z>(f_<X,Y>(x_X)))))(L x_e : L y_e : Saw(y,x))')



