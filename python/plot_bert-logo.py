get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import matplotlib.textpath
import matplotlib.patheffects as path_effects
import pygimli as pg

logo_path = matplotlib.textpath.TextPath((0,0), r'$\Omega$', size=1)
patch = matplotlib.patches.PathPatch(logo_path)

nodes = patch.get_verts() * 50
nodes2 = pg.utils.unique_rows(nodes) # remove duplicate nodes
poly = pg.Mesh(2)

for node in nodes:
    poly.createNode(node[0], node[1], 0.0)

for i in range(poly.nodeCount() - 1):
    poly.createEdge(poly.node(i), poly.node(i + 1))

poly.createEdge(poly.node(poly.nodeCount() - 1), poly.node(0))

tri = pg.TriangleWrapper(poly)
mesh = pg.Mesh(2)
tri.setSwitches('-pzeAfa5q31')
tri.generate(mesh)

data = []
for cell in mesh.cells():
    data.append(cell.center().x())

fig, ax = plt.subplots(figsize=(4,3))
ax.axis('off')
offset = -10
t = ax.text(1.5, offset, 'BERT', fontsize=50, fontweight='bold')
t.set_path_effects([path_effects.PathPatchEffect(offset=(3, -3), hatch='xxxx',
                                                 facecolor='gray', alpha=0.5),
                    path_effects.PathPatchEffect(edgecolor='white', linewidth=1,
                                                 facecolor='black')])
pg.show(mesh, data, axes=ax, cmap='RdBu', logScale=False, showLater=True)
pg.show(mesh, axes=ax)
ax.set_ylim(offset, mesh.ymax())

