# local hack to show latex figures 
# on a local sage, use view
from viewLatex import viewLatex

# creating an interval-poset
ip = TamariIntervalPoset(4,[(2,1),(3,1),(2,4),(3,4)])
ip

viewLatex(ip)

bt1 = BinaryTree([[None, [[], None]], None])
bt2 = BinaryTree([None, [[None, []], None]])
ip = TamariIntervalPosets.from_binary_trees(bt1,bt2)
ip

ip = bt1.tamari_interval(bt2)
ip

dw1 = DyckWord([1, 1, 0, 1, 0, 0, 1, 0])
dw2 = DyckWord([1, 1, 1, 0, 0, 1, 0, 0])
ip = TamariIntervalPosets.from_dyck_words(dw1,dw2)
ip

ip = dw1.tamari_interval(dw2)
ip

print ip.lower_binary_tree()
print ip.upper_binary_tree()

viewLatex(ip.lower_binary_tree())

viewLatex(ip.upper_binary_tree())

viewLatex(ip.lower_dyck_word())

viewLatex(ip.upper_dyck_word())

for bt in ip.binary_trees():
    print bt

for dw in ip.dyck_words():
    print dw

def left_product(ip1,ip2):
    size = ip1.size() + ip2.size()
    # Juxtaposition of ip1 and shifted ip2
    relations = list(ip1._cover_relations) + [(i+ip1.size(),j+ip1.size()) for (i,j) in ip2._cover_relations]
    # Increasing relations between ip1 and the first vertex of ip2
    relations+= [(i,ip1.size()+1) for i in ip1.increasing_roots()]
    return TamariIntervalPoset(size,relations)
    
def right_product(ip1, ip2):
    size = ip1.size() + ip2.size()
    # Juxtaposition of ip1 and shifted ip2
    relations = list(ip1._cover_relations) + [(i+ip1.size(),j+ip1.size()) for (i,j) in ip2._cover_relations]
    # First element: no extra decreasing relation
    yield TamariIntervalPoset(size,relations)
    for j in ip2.decreasing_roots():
        # Adding decreasing relations 1 by 1
        relations.append((j+ip1.size(),ip1.size()))
        yield TamariIntervalPoset(size,relations)
        
def composition(ip1,ip2):
    u = TamariIntervalPoset(1,[])
    left = left_product(ip1,u)
    for r in right_product(left,ip2):
        yield r

ip1 = TamariIntervalPoset(3,[(1,2),(3,2)])
ip2 = TamariIntervalPoset(4,[(2,3),(4,3)])
r = list(composition(ip1,ip2))
r

viewLatex(r)

def right_product_dim(ip1,ip2):
    size = ip1.size() + ip2.size()
    # Juxtaposition of ip1 and shifted ip2
    relations = list(ip1._cover_relations) + [(i+ip1.size(),j+ip1.size()) for (i,j) in ip2._cover_relations]
    for j in ip2.decreasing_roots():
        # Adding decreasing relations 1 by 1
        relations.append((j+ip1.size(),ip1.size()))
        yield TamariIntervalPoset(size,relations)
        
def mcomposition(ips):
    rights = ips[1:] # we take the list of intervals excpect the first one
    def compute_rights(rights): # we define a recursive method to compute the right part
        u = TamariIntervalPoset(1,[])
        if len(rights)==1:
            for r in right_product(u,rights[0]):
                yield r
        else:
            for r1 in compute_rights(rights[1:]):
                r1 = left_product(r1,rights[0])
                for r2 in right_product_dim(u,r1):
                    yield r2
    for r in compute_rights(rights):
        yield left_product(ips[0],r)

ip1 = TamariIntervalPoset(2,[(2,1)])
ip2 = TamariIntervalPoset(4,[(2,1),(4,3),(2,3)])
ip3 = ip1
r = list(mcomposition([ip1,ip2,ip3]))
r

viewLatex(r)



