reload_lamb()
#lang.set_system(lang.hk3_system)

get_ipython().run_cell_magic('lamb', 'reset', "||gray|| = lambda x_e : Gray(x)\n||cat|| = lambda x_e : Cat(x)\n||joanna|| = Joanna_e\n||isV|| = lambda f_<e,t> : f # 'is' is a python reserved word\n||a|| = lambda f_<e,t> : f")



r = gray * cat
r

r.tree()

r = joanna * (isV * (a * (gray * cat)))
r.tree()

get_ipython().run_cell_magic('lamb', '', '||inP|| = L x: L y: In(y,x)\n||fond|| = L x_e : L y_e : Fond(y,x)\n||of|| = L x_e : x\n||joe|| = Joe_e\n||texas|| = Texas_e')

r = joanna * (isV * (a * ((gray * cat) * (inP * texas))))
r

r = joanna * (isV * (a * (((gray * cat) * (inP * texas)) * (fond * (of * joe)))))
r

# show all the steps of each composition.  PM is currently implemented using function application underlyingly.
r[0].tree(derivations=True)





x = lang.te("L f_<e,t> : f")
y = lang.te("L x_e : Test(x)")

(x(y))



z = (x(y)).reduce()
z.derivation



reload_lamb()
lang.set_system(lang.hk3_system)

get_ipython().run_cell_magic('lamb', 'reset', '||gray|| = L x_e : Gray_<e,t>(x)\n||cat|| = L x_e : Cat_<e,t>(x)')

t = Tree("NP", ["gray", "cat"])
r = lang.hk3_system.compose(t)
#lang.hk3_system.expand_next(r)
#lang.hk3_system.expand_all(r)
r.tree()

lang.hk3_system.expand_next(r)
r.tree()

lang.hk3_system.expand_next(r)
r.tree()

r.paths()



