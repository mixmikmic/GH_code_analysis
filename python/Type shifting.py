reload_lamb()

get_ipython().run_cell_magic('lamb', '', 'gqlift = L x_e : L f_<e,t> : f(x)')

get_ipython().magic('lamb thecat = Iota x_e : Cat_<e,t>(x)')
gqlift(thecat).reduce_all()

get_ipython().run_cell_magic('lamb', '', '||gray|| = L x_e : Gray(x)\n||cat|| = L x_e : Cat(x)')

(gray * cat).tree()

pm_combinator = te("L f_<e,t> : L g_<e,t> : L x_e : f(x) & g(x)")
pm_combinator

system = lang.td_system.copy()
system.add_rule(lang.unary_factory(pm_combinator, "PM-shift", typeshift=True)) # add this as a typeshift
system.remove_rule("PM") # remove ordinary PM
system.typeshift = True # enable typeshifts in the composition system
lang.set_system(system) # set the new system as the default
system

r = (gray * cat)
r

r.tree()

get_ipython().run_cell_magic('lamb', '', '||every|| = L f_<e,t> : L g_<e,t> : Forall x_e : f(x) >> g(x)\n||doctor|| = L x_e : Doctor(x)\n||someone|| = L f_<e,t> : Exists x_e : Human(x) & f(x)\n||saw|| = L x_e : L y_e : Saw(y,x)\n||alfonso|| = Alfonso_e')

((every * doctor) * (saw * alfonso)).tree()

(saw * (every * doctor))

gq_lift_combinator = get_ipython().magic('te L f_<<e,t>,t> : L g_<e,<e,t>> : L x_e : f(L y_e : g(y)(x))')
gq_lift_combinator

gq_lift_combinator(someone.content).reduce_all()



system = lang.td_system.copy()
system.add_rule(lang.unary_factory(gq_lift_combinator, "gq-lift-trans", typeshift=True))
system.typeshift = True
lang.set_system(system)
system

(alfonso * (saw * someone))

(someone * (saw * (every * doctor)))

r = ((every * doctor) * (saw * someone))
r

r.tree()

r = (someone * (saw * (every * doctor)))
r

surface_shift_comb = te("L v_<e,<e,t>> : L f_<<e,t>,t> : L g_<<e,t>,t> : g(L y_e : f(L x_e : (v(x)(y))))")
inverse_shift_comb = te("L v_<e,<e,t>> : L f_<<e,t>,t> : L g_<<e,t>,t> : f(L x_e : g(L y_e : (v(x)(y))))")

inverse_shift_comb(saw.content).reduce_all()

system = lang.td_system.copy()
system.add_rule(lang.unary_factory(surface_shift_comb, "surface", typeshift=True))
system.add_rule(lang.unary_factory(inverse_shift_comb, "inverse", typeshift=True))
system.typeshift = True
lang.set_system(system)
system

r = (someone * ((every * doctor) * saw))
r

r[1].tree()

gq_lift_combinator = te("L f_<<e,t>,t> : L g_<e,<e,t>> : L x_e : f(L y_e : g(y)(x))")
gq_lift_combinator2 = te("L f_<<e,t>,t> : L g_<e,<e,t>> : L h_<<e,t>,t> : f(L y_e : h(L x_e : g(y)(x)))")

gq_lift_combinator2.type

system = lang.td_system.copy()
system.add_rule(lang.unary_factory(gq_lift_combinator, "gq-lift-trans", typeshift=True))
system.add_rule(lang.unary_factory(gq_lift_combinator2, "gq-lift2-trans", typeshift=True))
system.typeshift = True
lang.set_system(system)
system

r = (someone * ((every * doctor) * saw))
r

r.tree()



