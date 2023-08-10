from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()
from aiida.orm import load_node
from IPython.display import Image
import markdown

def generate_graph(pk):
    get_ipython().system('verdi graph generate {pk}')
    get_ipython().system('dot -Tpng {pk}.dot -o {pk}.png')
    return "{}.png".format(pk)

get_ipython().system('verdi calculation list -a')

generate_graph(4811)

Image(filename='4811.png')

calc = load_node(4811)
calc.get_inputs(also_labels=True)

structure = calc.inp.structure

generate_graph(structure.pk)
Image(filename=str(structure.pk)+'.png')

structure.get_inputs(also_labels=True)

get_ipython().system('verdi data structure list')

generate_graph(4684)
Image(filename='4684.png')

calc = load_node(4685)
calc.out.output_parameters.get_dict()

get_ipython().system('verdi work tree 4685')



