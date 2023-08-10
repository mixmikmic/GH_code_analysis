from support.interbaseinterval import InterbaseInterval
from IPython.display import display, Markdown

iis = [
    InterbaseInterval(0,3),
    InterbaseInterval(0,2),
    InterbaseInterval(1,3),
    InterbaseInterval(0,1),
    InterbaseInterval(1,2),
    InterbaseInterval(2,3),
    InterbaseInterval(1,1),
    InterbaseInterval(2,2),
    ]

INTERVAL_METHODS = "abuts coincides_with encloses intersects overlap overlaps".split()

def pairwise_apply(iis, method_name):
    """return matrix of method_name applied to iis pairs"""
    m = getattr(InterbaseInterval, method_name)
    return [[m(r,c) for c in iis] for r in iis]

def pairwise_apply_md(iis, method_name):
    """return markdown table of pairwise_apply"""
    tbl = [["r.{}(c)".format(method_name)] + [str(ii) for ii in iis]]
    tbl += [["-"] * len(tbl[0])]
    for ii, row in zip(iis, pairwise_apply(iis, method_name)):
        tbl += [[str(ii)] + row]
    tbl_str = "\n".join(["|".join(map(str,row)) for row in tbl])
    return Markdown("# "+method_name + "\n" + tbl_str)

for mn in INTERVAL_METHODS:
    display(pairwise_apply_md(iis, mn))

