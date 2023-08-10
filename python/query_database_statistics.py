from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()
from aiida.orm import load_node, Node, Group, Computer, User
from aiida.orm.querybuilder import QueryBuilder
from IPython.display import Image

def generate_query_graph(qh, out_file_name):

    def draw_vertice_settings(idx, vertice, **kwargs):
        """
        Returns a string with all infos needed in a .dot file  to define a node of a graph.
        :param node:
        :param kwargs: Additional key-value pairs to be added to the returned string
        :return: a string
        """
        if 'calculation' in vertice['type']:
            shape = "shape=polygon,sides=4"
        elif 'code' in vertice['type']:
            shape = "shape=diamond"
        else:
            shape = "shape=ellipse"
        filters = kwargs.pop('filters', None)
        additional_string = ""
        if filters:
            additional_string += '\nFilters:'
            for k,v in filters.items():
                additional_string += "\n   {} : {}".format(k,v)


        label_string = " ('{}')".format(vertice['tag'])

        labelstring = 'label="{} {}{}"'.format(
            vertice['type'], #.split('.')[-2] or 'Node',
            label_string,
            additional_string)
        #~ return "N{} [{},{}{}];".format(idx, shape, labelstring,
        return "{} [{},{}];".format(vertice['tag'], shape, labelstring)
    nodes = {v['tag']:draw_vertice_settings(idx, v, filters=qh['filters'][v['tag']]) for idx, v in enumerate(qh['path'])}
    links = [(v['tag'], v['joining_value'], v['joining_keyword']) for v in qh['path'][1:]]

    with open('temp.dot','w') as fout:
        fout.write("digraph G {\n")
        for l in links:
            fout.write('    {} -> {} [label=" {}"];\n'.format(*l))
        for _, n_values in nodes.items():
            fout.write("    {}\n".format(n_values))

        fout.write("}\n")
    import os
    os.system('dot temp.dot -Tpng -o {}'.format(out_file_name))

print "My database contains:"
for cls in (User, Computer, Group, Node):
    qb = QueryBuilder()
    qb.append(cls)
    count = qb.count()
    print "{:>5}  {}s".format(count, cls.__name__)

for cls in (Node, Group):
    print '\n', 'Subclasses of {}:'.format(cls.__name__)
    qb1 = QueryBuilder()
    qb1.append(cls, project='type')
    distinct_types, = zip(*qb1.distinct().all()) # Getting all distinct types
    # Iterating through distinct types:
    for dtype in sorted(distinct_types):
        qb2 = QueryBuilder()
        qb2.append(cls, filters={'type':dtype})
        subcls_count = qb2.count()
        print '   {:<15} | {:<4}'.format(dtype.strip('.').split('.')[-1] or "N/A", subcls_count)

# Here I query the number of links:
qb1 = QueryBuilder()
qb1.append(Node, tag='n1')
qb1.append(Node, output_of='n1')
link_count = qb1.count()
print '\nThe number of links in my database is: {}'.format(link_count)

generate_query_graph(qb1.get_json_compatible_queryhelp(), 'query-statistics-1.png')
Image(filename='query-statistics-1.png')

# Here I query the number of distinct paths:
qb2 = QueryBuilder()
qb2.append(Node, tag='n1')
qb2.append(Node, descendant_of='n1')
path_count = qb2.count()
print '\nThe number of distinct paths in my database is: {}'.format(path_count)

generate_query_graph(qb2.get_json_compatible_queryhelp(), 'query-statistics-2.png')
Image(filename='query-statistics-2.png')



