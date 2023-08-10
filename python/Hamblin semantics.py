reload_lamb()
get_ipython().magic('lambctl reset')

type_e = types.type_e
type_t = types.type_t
type_n = types.type_n
type_s = types.BasicType("s")
ts = meta.get_type_system()
ts.add_atomic(type_s)
ts

def hamblinize_te(te):
    """Hamblinize a single lexical item.  Do so by building a singleton set out of it."""
    if meta.get_type_system().unify(te.type, meta.tp("{X}")): 
        # assume this item is already hamblinized
        return te
    elif meta.get_type_system().unify(te.type, meta.tp("<{X},Y>")): 
        # heuristic: if function whose domain is a set of some sort, assume that this is a Hamblin operator.
        # may miss cases.  Better to just run this on content items...
        return te
    # wrap the content of the lexical item as a singleton set.
    return meta.ListedSet([te])    

#continuize_lex(cat1.content)
lamb.parsing.eq_transforms["hamblin"] = hamblinize_te

hamblinize_te(te("L x: Cat(x)"))

get_ipython().run_cell_magic('lamb', '', "||cat|| =<hamblin> L x_e : L w_s : Cat(w,x)\n||gray|| =<hamblin> L x_e : L w_s : Gray(w,x)\n||john|| =<hamblin> John_e\nx =<hamblin> L y_e : y\n||test|| = L x_e : Test(x) # don't hamblinize this")



pfa_combinator = get_ipython().magic('te L f_{<X,Y>} : L a_{X} : Set x_Y : Exists f1_<X,Y> : (Exists a1_X : (f1 << f & a1 << a) & x <=> f1(a1))')

def pfa_combinator2(funtype, argtype):
    return te("L f_{%s} : L a_{%s} : Set x_%s : Exists f1_%s : (Exists a1_%s : (f1 << f & a1 << a) & x <=> f1(a1))" % (repr(funtype), repr(argtype), repr(funtype.right), repr(funtype), repr(argtype)))

pfa_combinator2(tp("<e,<s,t>>"), types.type_e)
pfa_combinator

def pfa_worstcase(fun, arg):
    ts = meta.get_type_system()
    if not (isinstance(fun.type, types.SetType) 
            and isinstance(arg.type, types.SetType)):
        raise types.TypeMismatch(fun, arg, "Pointwise Function Application")
    if not (ts.fun_arg_check_types_bool(fun.type.content_type, arg.type.content_type)):
        raise types.TypeMismatch(fun, arg, "Pointwise Function Application")
    return pfa_combinator(fun.type.content_type, arg.type.content_type)(fun)(arg)

system = lang.td_system.copy()
#system.add_rule(lang.binary_factory(pfa_worstcase, "PFA"))
system.add_binary_rule(pfa_combinator, "PFA")
lang.set_system(system)
system



john * cat

r = te("x_e << (Set y_e : Test_<e,t>(y))")
r.reduce_all().derivation

def pfa_listed(fun, arg):
    result = list()
    for felem in fun.args:
        for aelem in arg.args:
            result.append(felem(aelem))
    return meta.ListedSet(result)

def pfa_general(fun, arg):
    ts = meta.get_type_system()
    general = pfa_combinator(fun)(arg) # do this for type checking
    if isinstance(fun, meta.ListedSet) and isinstance(arg, meta.ListedSet):
        return pfa_listed(fun, arg)
    else:
        return general.reduce_all()
    
system = lang.td_system.copy()
system.add_binary_rule_uncurried(pfa_general, "PFA")
lang.set_system(system)
system

john * cat



get_ipython().run_cell_magic('lamb', '', "## To simplify, let's take there to only be three human-like entities in the universe.\n||who|| = {John_e, Mary_e, Sue_e}\n||saw|| =<hamblin> L x_e : L y_e : L w_s : Saw(w,y,x)")

(cat * who).tree()

john * (saw * who)

(john * (saw * who)).tree()

who * (saw * john)

get_ipython().run_cell_magic('lamb', '', '||HExists|| = L p_{<s,t>} : {(Lambda w_s  : Exists q_<s,t> : q(w) & (q << p))}\n||HForall|| = L p_{<s,t>} : {(Lambda w_s  : Forall q_<s,t> : q(w) >> (q << p))}')

HExists * (who * (saw * john))

HExists * (john * (saw * who))

(HExists * (john * (saw * who))).tree()

get_ipython().run_cell_magic('lamb', '', '||who|| = Set x_e : Human(x)')

(cat * who).tree()

john * (saw * who)

who * (saw * john)

HExists * (who * (saw * john))

raise Exception("Prevent run-all from working")

def hamblinize_item(item):
    """Hamblinize a single lexical item.  Do so by building a singleton set out of it."""
    if meta.ts_compatible(item.type, meta.tp("{?}")): #isinstance(item.type, types.SetType):
        # assume this item is already hamblinized
        return item
    elif meta.ts_compatible(item.type, meta.tp("<{?},?>")): #item.type.functional() and isinstance(item.type.left, types.SetType):
        # heuristic: if function whose domain is a set of some sort, assume that this is a Hamblin operator.
        # may miss cases.  Better to just run this on content items...
        return item
    new_i = item.copy()
    # wrap the content of the lexical item as a singleton set.
    new_i.content = meta.ListedSet([item.content])
    return new_i

# in the following two magics, variables that are not lexical items are ignored.  To change this, modify the else case above.
def h_magic(self, accum):
    """Hamblinize the accumulated definitions from a single cell, as a post-processing step"""
    new_accum = lamb.magics.process_items(hamblinize_item, accum)[0]
    for k in new_accum.keys():
        self.env[k] = new_accum[k]
    return new_accum

def h_magic_env(self):
    """Hamblinize the entire env"""
    self.env = lamb.magics.process_items(hamblinize_item, self.env)[0] # hamblinize every variable
    self.shell.push(self.env) # export the new variables to the interactive shell
    return parsing.latex_output(self.env, self.env)

lamb.magics.LambMagics.specials_post["hamblinize"] = h_magic
lamb.magics.LambMagics.specials["hamblinize_all"] = h_magic_env

get_ipython().run_cell_magic('lamb', 'hamblinize', '||cat|| = L x_e : L w_s : Cat(w,x)\n||gray|| = L x_e : L w_s : Gray(w,x)\n||john|| = J_e\nx = L y_e : y')

get_ipython().run_cell_magic('lamb', '', '||test|| = L x_e : Test(x)')

get_ipython().magic('lambctl hamblinize_all')



