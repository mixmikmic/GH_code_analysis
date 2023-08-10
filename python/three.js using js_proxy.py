from IPython import display
load_three = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r70/three.min.js">
</script>
"""
display.display(display.HTML(load_three))

# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")
    
from jp_gene_viz import js_proxy
js_proxy.load_javascript_support()

w = js_proxy.ProxyWidget()

# Some shortcut names for proxy references for convenience:
# The global window namespace:
window = w.window()
# The jQuery element for the widget:
element = w.element()
# The THREE module object:
THREE = window.THREE
# The emulation of the JS "new" keyword.
new = w.save_new

scene = new("scene", THREE.Scene, [])
camera = new("camera", THREE.PerspectiveCamera, [75, 1.0, 1, 10000])
w(camera.position._set("z", 500))
geometry = new("geometry", THREE.BoxGeometry, [200, 200, 200])
material = new("material", THREE.MeshBasicMaterial, [{"color": 0xff0000, "wireframe": True } ])
mesh = new("mesh", THREE.Mesh, [geometry, material])
w(scene.add(mesh))
renderer = new("renderer", THREE.WebGLRenderer, [])
w(renderer.setSize(300, 300))
w(element.append(renderer.domElement))
do_render = w(renderer.render(scene, camera))

# send the buffered commands
json_sent = w.flush()

# show the 3d scene.
display.display(w)

# rotate the cube using a busy-loop blocking the interpreter.
import time

def make_it_rotate():
    for i in xrange(100):
        time.sleep(0.1)
        w(mesh.rotation._set("x", i/10.0)._set("y", i/5.0))
        w(do_render)
        w.flush()

make_it_rotate()

# rotate the cube using a requestAnimationFrame callback
# this doesn't block the interpreter.
requestAnimationFrame = window.requestAnimationFrame
rotation = {"x": 1.1, "y": 2.2, "count": 0}

def animation_frame(data=None, arguments=None):
    rotation["count"] += 1
    if rotation["count"] > 100000:
        return # stop animation
    rotation["x"] += 0.01
    rotation["y"] += 0.02
    w(mesh.rotation._set("x", rotation["x"])._set("y", rotation["y"]))
    w(do_render)
    w(requestAnimationFrame(animation_callback))
    w.flush()

# set up the js-->python callback interface
animation_callback = w.callback(animation_frame, data=None)

# start the animation sequence
animation_frame()

# do some calculation during the animation
12 + 90



