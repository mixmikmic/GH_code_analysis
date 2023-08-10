get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import sys
sys.path.insert(0, '../python-torchfile')
import torchfile

t = torchfile.load('vrn-unguided.t7')

ln = 0
list_file = open("list.txt", "w")

def lst(tobj, indent, f):
    global ln
    for module in tobj['modules']:
        f.write("{}\t{}{}\t{}\n".format(ln,
                                        "    " * indent,
                                        module._typename,
                                        module['weight'].shape 
                                            if 'weight' in module.__dir__() 
                                            else ''))
        ln += 1
        if 'modules' in module.__dir__():
            lst(module, indent + 1, f)

lst(t, 0, list_file)
list_file.close()

def getr(tobj, ln):
    for module in tobj['modules']:
        if ln == 0:
            return module, 0
        ln -= 1
        if 'modules' in module.__dir__():
            got, ln = getr(module, ln)
            if got is not None:
                return got, 0

    return None, ln

def get(tobj, ln):
    tout, _ = getr(tobj, ln)
    return tout

def info(module):
    print("{}".format(module._typename))
    for prop in module.__dir__():
        value = module[prop]
        if value.__class__ == np.ndarray:
            value = value.shape
        if value.__class__ == list and len(value) > 0:
            value = value[0].shape
        print("{}\t{}".format(prop, value))

info(get(t, 0))

info(get(t, 1))

