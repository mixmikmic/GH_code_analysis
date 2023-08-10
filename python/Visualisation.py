from apt_importers import *

pos = read_pos('../example-data/voldata.pos')
ions, rrngs = read_rrng('../example-data/rangefile.rrng')
lpos = label_ions(pos,rrngs)
dpos = deconvolve(lpos)

def volvis(pos, size=2, alpha=1):
    """Displays a 3D point cloud in an OpenGL viewer window.
    If points are not labelled with colours, point brightness
    is determined by Da values (higher = whiter)"""
    from vispy import app,scene,mpl_plot
    import numpy as np
    import sys
    import matplotlib.colors as cols
    import re
    
    canvas = scene.SceneCanvas('APT Volume',keys='interactive')
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up='z')
    
    cpos = pos.loc[:,['x','y','z']].values
    if 'colour' in pos.columns:
        colours = np.asarray(list(pos.colour.apply(cols.hex2color)))
    else:
        Dapc = lpos.Da.values / lpos.Da.max()
        colours = np.array(zip(Dapc,Dapc,Dapc))
    if alpha is not 1:
        np.hstack([colours, np.array([0.5] * len(colours))[...,None]])
    
    p1 = scene.visuals.Markers()
    p1.set_data(cpos, face_color=colours, edge_width=0, size=size)

    view.add(p1)
    
    # make legend
    ions = []
    cs = []
    for g,d in pos.groupby('colour'):
        ions.append(re.sub(r':1?|\s?','',d.comp.iloc[0]))
        cs.append(cols.hex2color(g))
    ions = np.array(ions)
    cs = np.asarray(cs)

    pts = np.array([[20] * len(ions), np.linspace(20,20*len(ions), len(ions))]).T
    tpts = np.array([[30] * len(ions), np.linspace(20,20*len(ions), len(ions))]).T
    
    legb = scene.widgets.ViewBox(parent=view, border_color='red', bgcolor='k')
    legb.pos = 0,0
    legb.size = 100,20*len(ions)+20
    
    leg = scene.visuals.Markers()
    leg.set_data(pts, face_color=cs)
    legb.add(leg)
    
    legt = scene.visuals.Text(text=ions,pos=tpts,color='white', anchor_x='left', anchor_y='center', font_size=10)
    
    legb.add(legt)
    
    # show viewer
    canvas.show()
    if sys.flags.interactive == 0: 
        app.run()

volvis(dpos,alpha=0.6)

volvis(dpos.loc[dpos.element=='Na',:])

