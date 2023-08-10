from pythreejs import *
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Text
from traitlets import link, dlink

nx, ny = (20, 20)
xmax=1
x = np.linspace(-xmax, xmax, nx)
y = np.linspace(-xmax, xmax, ny)
xx, yy = np.meshgrid(x, y)
z = xx ** 2 - yy ** 2
#z[6,1] = float('nan')
surf_g = SurfaceGeometry(z=list(z[::-1].flat), 
                         width=2 * xmax,
                         height=2 * xmax,
                         width_segments=nx - 1,
                         height_segments=ny - 1)

surf = Mesh(geometry=surf_g, material=LambertMaterial(map=height_texture(z[::-1], 'YlGnBu_r')))
surfgrid = SurfaceGrid(geometry=surf_g, material=LineBasicMaterial(color='black'))
hover_point = Mesh(geometry=SphereGeometry(radius=0.05), material=LambertMaterial(color='hotpink'))
scene = Scene(children=[surf, surfgrid, hover_point, AmbientLight(color='#777777')])
c = PerspectiveCamera(position=[0, 3, 3], up=[0, 0, 1], 
                      children=[DirectionalLight(color='white', position=[3, 5, 1], intensity=0.6)])
click_picker = Picker(root=surf, event='dblclick')
hover_picker = Picker(root=surf, event='mousemove')
renderer = Renderer(camera=c, scene = scene, controls=[OrbitControls(controlling=c), click_picker, hover_picker])

def f(change):
    value = change['new']
    print('Clicked on %s' % value)
    point = Mesh(geometry=SphereGeometry(radius=0.05), 
                 material=LambertMaterial(color='red'),
                 position=value)
    scene.children = list(scene.children) + [point]

click_picker.observe(f, names=['point'])

link((hover_point, 'position'), (hover_picker, 'point'))

h = HTML()
def g(change):
    h.value = 'Pink point at (%.3f, %.3f, %.3f)' % tuple(change['new'])
g({'new': hover_point.position})
hover_picker.observe(g, names=['point'])
display(h)
display(renderer)

# when we change the z values of the geometry, we need to also change the height map
surf_g.z = list((-z[::-1]).flat)
surf.material.map = height_texture(-z[::-1])

f = """
function f(origu,origv) {
    // scale u and v to the ranges I want: [0, 2*pi]
    var u = 2 * Math.PI * origu;
    var v = 2 * Math.PI * origv;
    
    var x = Math.sin(u);
    var y = Math.cos(v);
    var z = Math.cos(u + v);
    
    return new THREE.Vector3(x, y, z)
}
"""
surf_g = ParametricGeometry(func=f);

surf = Mesh(geometry=surf_g, material=LambertMaterial(color='green', side='FrontSide'))
surf2 = Mesh(geometry=surf_g, material=LambertMaterial(color='yellow', side='BackSide'))
scene = Scene(children=[surf, surf2, AmbientLight(color='#777777')])
c = PerspectiveCamera(position=[5, 5, 3], up=[0, 0, 1],
                      children=[DirectionalLight(color='white',
                                                 position=[3, 5, 1],
                                                 intensity=0.6)])
renderer = Renderer(camera=c, scene=scene, controls=[OrbitControls(controlling=c)])
display(renderer)

from ipywidgets import FloatSlider, HBox, VBox

x_slider, y_slider, z_slider = (FloatSlider(description='x', min=-10.0, max=10.0, orientation='vertical'),
                                FloatSlider(description='y', min=-10.0, max=10.0, orientation='vertical'),
                                FloatSlider(description='z', min=-10.0, max=10.0, orientation='vertical'))

shape_slider_x = FloatSlider(description='Shape x', min=0, max=1.5, step=0.01, continuous_update=False)
shape_slider_y = FloatSlider(description='Shape y', min=0, max=1.5, step=0.01, continuous_update=False)
shape_slider_z = FloatSlider(description='Shape z', min=0, max=1.5, step=0.01, continuous_update=False)

def update(change):
    c.position = [x_slider.value, y_slider.value, z_slider.value]
    
x_slider.observe(update, names=['value'])
y_slider.observe(update, names=['value'])
z_slider.observe(update, names=['value'])

def update_shape(change):
    surf_g.func = """
    function f(origu, origv) {
        // scale u and v to the ranges I want: [0, 2*pi]
        var u = 2 * Math.PI * origu;
        var v = 2 * Math.PI * origv;

        var x = (1 + 0.5 * %s * Math.sin(u)) * Math.sin(u);
        var y = (1 + 0.5 * %s * Math.sin(u)) * Math.cos(v);
        var z = (1 + 0.5 * %s * Math.sin(u)) * Math.cos(u+v);

        return new THREE.Vector3(x,y,z)
    }
    """ % (shape_slider_x.value, shape_slider_y.value, shape_slider_z.value)
    
shape_slider_x.observe(update_shape, names=['value'])
shape_slider_y.observe(update_shape, names=['value'])
shape_slider_z.observe(update_shape, names=['value'])

VBox([HBox([renderer, x_slider, y_slider, z_slider]), shape_slider_x, shape_slider_y, shape_slider_z])

