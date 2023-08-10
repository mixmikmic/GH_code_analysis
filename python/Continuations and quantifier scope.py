import imp
imp.reload(lamb.parsing)
reload_lamb()

get_ipython().run_cell_magic('lamb', '', '||raining1|| = Raining_t # reminder: hit shift-enter to run\n||raining2|| = L f_<t,t> : f(Raining_t)\n||john1|| = John_e\n||john2|| = L f_<e,t> : f(John_e)\n||cat1|| = L x_e : Cat(x)\n||cat2|| = L f_<<e,t>,t> : f(L x_e : Cat(x))')

get_ipython().run_cell_magic('lamb', '', 'econt = L x_e : L f_<e,t> : f(x)')

econt(john1.content).reduce_all() # we get back the content of john2

get_ipython().run_cell_magic('lamb', '', 'continuize = L x_X : L f_<X,t> : f(x)')

def continuize_lex(te):
    new_te = continuize(te).reduce_all()
    return new_te

#continuize_lex(cat1.content)
lamb.parsing.eq_transforms["cont"] = continuize_lex

get_ipython().run_cell_magic('lamb', '', '||cat2|| =<cont> L x_e : Cat(x)\n||dance2|| =<cont> L x_e : Dance(x)\n||john2|| =<cont> John_e\n||the|| =<cont> L f_<e,t> : Iota x_e : f(x)')

get_ipython().run_cell_magic('lamb', '', 'cfaet = L f_<<<e,t>,t>,t> : L arg_<<e,t>,t> : L abar_<t,t> : f(L b_<e,t> : arg(L c_e : abar(b(c))))')

(cfaet(dance2.content)(john2.content)).reduce_all().derivation

get_ipython().run_cell_magic('lamb', '', 'contapply = L f_Z : L arg_Z1 : L abar_Z2 : f(L b_<X,X1> : arg(L c_X : abar(b_<X,X1>(c_X))))')

contapply(cat2.content)(john2.content)

contapply(cat2.content)(john2.content).reduce_all()

contapply(continuize(cat1.content))(continuize(john1.content)).reduce_all()

contapply(the.content)(cat2.content)

contapply(the.content)(cat2.content).reduce_all()

system = lang.td_system.copy()
system.remove_rule("FA")
system.remove_rule("PA")
system.remove_rule("PM")
#system.add_rule(ca_op)
system.add_binary_rule(contapply, "CA")
lang.set_system(system)
system

(john2 * dance2).tree()

get_ipython().run_cell_magic('lamb', '', '||saw|| =<cont> L y_e : L x_e : Saw(x,y)\n||mary|| =<cont> Mary_e')

john2 * (saw * mary)

(john2 * (saw * mary)).tree()

get_ipython().run_cell_magic('lamb', '', '||someone|| = L xbar_<e,t> : Exists x_e : xbar(x)\n||everyone|| = L xbar_<e,t> : Forall x_e : xbar(x)\n||everyone0|| =<cont> L f_<e,t> :  Forall x_e : f(x)')

everyone * (saw * mary)

everyone0 * (saw * mary)

(saw * everyone0)

mary * (saw * everyone)

everyone * (saw * someone)

get_ipython().run_cell_magic('lamb', '', 'contapply2 = L f_Z : L arg_Z1 : L abar_Z2 : arg(L c_X : f(L b_<X,X1>: abar(b_<X,X1>(c_X))))')

system = lang.td_system.copy()
system.remove_rule("FA")
system.remove_rule("PA")
system.remove_rule("PM")
system.add_binary_rule(contapply, "CA")
system.add_binary_rule(contapply2, "CA2")
lang.set_system(system)
system

everyone * (saw * mary)

everyone * (saw * someone)

(someone * (saw * everyone))

(someone * (saw * everyone))[1].tree()

get_ipython().run_cell_magic('lamb', '', '||every|| = L pbar_<<<e,t>,t>,t> : L xbar_<e,t> : pbar(L f_<e,t> : Forall x_e : (f(x) >> xbar(x)))')

system = lang.td_system.copy()
#system.remove_rule("FA")
system.remove_rule("PA")
system.remove_rule("PM")
system.add_binary_rule(contapply, "CA")
system.add_binary_rule(contapply2, "CA2")
lang.set_system(system)

def tfilter_fun(i):
    return (i.type == lang.tp("<<t,t>,t>"))

tfilter = lang.CRFilter("S-filter", tfilter_fun)

(every * cat2).tree()

r = (every * cat2) * (saw * someone)
tfilter(r)

r[1].tree()

get_ipython().run_cell_magic('lamb', '', '||Disrupt|| = L s_<<t,t>,t> : L abar_<t,t> : abar(s(L p_t : p))')

get_ipython().run_cell_magic('lamb', '', '||iift|| =<cont> L p_t : ~p ')

tfilter(Disrupt * (someone * dance2))

tfilter(someone * dance2)

tfilter(iift * (someone * dance2))

r2 = tfilter(iift * (Disrupt * (someone * dance2)))
r2

get_ipython().run_cell_magic('lamb', '', '||sneg|| = L f_<<t,t>,t> : ~ f(L p_t : p)')

tfilter(sneg * (someone * dance2))

get_ipython().run_cell_magic('lamb', '', '||no|| = L pbar_<<<e,t>,t>,t> : L xbar_<e,t> : pbar(L f_<e,t> : (~ (Exists x_e : (f(x) & xbar(x)))))\n||a|| = L pbar_<<<e,t>,t>,t> : L xbar_<e,t> : pbar(L f_<e,t> : (Exists x_e : (f(x) & xbar(x))))\n||fromP|| =<cont> L x_e : L f_<e,t> : L y_e : f(y) & From(y,x)\n||france|| =<cont> France_e\n||fcountry|| =<cont> L x_e : ForeignCountry(x)')

tfilter((no * (cat2 * (fromP * france))) * dance2)

r = tfilter((no * (cat2 * (fromP * (a * fcountry)))) * dance2)
r

r[0].tree()

get_ipython().run_cell_magic('lamb', '', '||a|| = L dbar_<<<e,t>,e>,t> : Exists f_<<e,t>,e> : dbar(f)\n||no|| = L dbar_<<<e,t>,e>,t> : ~(Exists f_<<e,t>,e> : dbar(f))\n||every|| = L dbar_<<<e,t>,e>,t> : (Forall f_<<e,t>,e> : dbar(f))')

every * cat2

(every * cat2)[0].tree(derivations=True)

tfilter((no * (cat2 * (fromP * france))) * dance2)

r = tfilter((no * (cat2 * (fromP * (a * fcountry)))) * dance2)
r

r[0].tree()

r2 = tfilter((no * (cat2 * (fromP * (a * fcountry)))) * (saw * everyone))
r2 



