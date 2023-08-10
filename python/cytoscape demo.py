# Add parent directory to the path so the imports will work without an install.
import sys
sys.path.append("..")
from jp_gene_viz import jp_cytoscape
jp_cytoscape.load_javascript_support(True)

# Display a cytoscape widget.
cy = jp_cytoscape.CytoscapeWidget()
js = cy.js()
cy

# remove the default node and make sure scrolling stays in sync
cy.send(js.fix())

# setup a fancy style
stylecmd = (js.style()        
    .selector('node')
      .css({
        'content': 'data(name)',
        'text-valign': 'center',
        'color': 'white',
        'text-outline-width': 2,
        'text-outline-color': '#888',
            "background-color": "cyan"
      })
    .selector('edge')
      .css({
        'target-arrow-shape': 'triangle'
      })
    .selector(':selected')
      .css({
        'background-color': 'black',
        'line-color': 'black',
        'target-arrow-color': 'black',
        'source-arrow-color': 'black'
      })
    .selector('.faded')
      .css({
        'opacity': 0.25,
        'text-opacity': 0
      }).update()
           )
cy.send(stylecmd)

# Load a graph and adjust the sizes and layout
people = ["jerry", "elaine", "kramer", "george"]
#from string import capitalize
js = cy.js()

for person in people:
    cmd = js.add({"group": "nodes", "data": {"id": person[0], "name": person.capitalize()}})
    cy.send(cmd)
for p1 in people:
    for p2 in people:
        cmd = js.add({"group": "edges", "data": { "source": p1[0], "target": p2[0]}})
        cy.send(cmd)
cy.width = "500px"
cy.height = "500px"
cy.send(js.resize())
cy.send(js.makeLayout({"name": "grid", "padding": 10}).run())
cy.send(js.forceRender())

# make elaine's node pink
cy.send(js.DOLLAR("#e").style("background-color", "pink").style("color", "yellow"))



