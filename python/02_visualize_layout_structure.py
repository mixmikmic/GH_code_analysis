# For parsing XML
from lxml import etree as et

# For formatting the content
import textwrap
import codecs

# For testing
from IPython.display import Image

graphviz_installed = get_ipython().getoutput('which dot')
if graphviz_installed == '':
    print "Graphviz/DOT not found. Exiting ..."
    quit()
else:
    print "\nGraphviz/DOT found ..."

layoutfile = '/Users/tuomo/Dropbox/Work/HER/00_annotated_data/2014-fin-annual_report-layout-1.xml' # Layout layer

def parse_xml(layoutfile):

    layout_root = {} # Set up a dictionary for the layout root
    layout_leafs = {} # Set up a dictionary for the layout leafs
    layout_chunks = {} # Set up a dictionary for the layout chunks
    
    layout_xmlroot = et.parse(layoutfile).getroot()
        
    for root in layout_xmlroot.xpath('.//layout-structure/layout-root'):
        layout_root[root.attrib['id']] = '"' + root.attrib['id'] + '" [fontcolor="black", fontsize="12.0", shape="box"];\n'
    
    if layout_xmlroot.find('.//realization') is not None: # Check if realization information is present
        for leaf in layout_xmlroot.xpath('.//layout-structure//layout-leaf'):
            leaf_xpath = './/realization/*[contains(@xref, "' + leaf.attrib['xref'] + '")]'
            for leaf_real in layout_xmlroot.xpath('%s' % leaf_xpath):
                if leaf_real.tag == 'text':
                    layout_leafs[leaf.attrib['xref']] = '"' + leaf.attrib['xref'] + '" [fontcolor="black", fontsize="10.0", shape="box"];\n'
                if leaf_real.tag == 'graphics':
                    layout_leafs[leaf.attrib['xref']] = '"' + leaf.attrib['xref'] + '" [fontcolor="black", fontsize="10.0", style="filled", fillcolor="burlywood2", shape="box"];\n'
                else:
                    continue
    else:
        for leaf in layout_xmlroot.xpath('.//layout-structure//layout-leaf'):
            layout_leafs[leaf.attrib['xref']] = '"' + leaf.attrib['xref'] + '" [fontcolor="black", fontsize="10.0", shape="box"];\n'
        
    for chunk in layout_xmlroot.xpath('.//layout-structure//layout-chunk'):
        layout_chunks[chunk.attrib['id']] = '"' + chunk.attrib['id'] + '" [fontcolor="black", fontsize="10.0", shape="box"];\n'
        
    return layout_xmlroot, layout_root, layout_leafs, layout_chunks

layout_xmlroot, layout_root, layout_leafs, layout_chunks = parse_xml(layoutfile)

graph = codecs.open('layout_graph.gv', 'w', 'utf-8')

begin_graph = 'graph "layout_graph" { graph [rankdir="BT"];\n'
terminate_graph = '}\n'

# Write DOT graph preamble
graph.write(begin_graph)

# Add layout root to the graph:
for root, root_node in layout_root.items():
    graph.write(root_node)

# Add layout leafs and their edges to the graph
for leaf, leaf_nodes in layout_leafs.items():
    graph.write(leaf_nodes)
    leaf_parent_xpath = './/layout-structure//layout-leaf[@xref="' + leaf + '"]/../@id'
    for leaf_parent in layout_xmlroot.xpath('%s' % leaf_parent_xpath):
        l_edge = '"' + leaf + '"' + ' -- ' + '"' + leaf_parent + '";\n'
        graph.write(l_edge)

# Add layout chunks and their edges to the graph
for chunk, chunk_nodes in layout_chunks.items():
    graph.write(chunk_nodes)
    chunk_parent_xpath = './/layout-structure//layout-chunk[@id="' + chunk + '"]/../@id'
    for chunk_parent in layout_xmlroot.xpath('%s' % chunk_parent_xpath):
        c_edge = '"' + chunk + '"' + ' -- ' + '"' + chunk_parent + '";\n'
        graph.write(c_edge)

# Terminate the graph
graph.write(terminate_graph)        

graph.close()

get_ipython().system('dot -Tpng layout_graph.gv > layout_graph.png')

Image('layout_graph.png')

