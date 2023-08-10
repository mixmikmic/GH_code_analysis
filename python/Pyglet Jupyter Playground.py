get_ipython().magic('gui pyglet')

import pyglet

window = pyglet.window.Window(display=None)
window.on_close = lambda:window.close()
label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

def draw_triangle():
    pyglet.gl.glBegin(pyglet.gl.GL_TRIANGLES)
    for p in [(20,30), (200,100), (100,200)]:
        pyglet.gl.glVertex3f(p[0], p[1],0)  # draw each vertex
    pyglet.gl.glEnd()



for _ in range(200):
    window.clear()
    window.switch_to()
    window.dispatch_events()

    label.draw()
    draw_triangle()
    
    window.flip()





