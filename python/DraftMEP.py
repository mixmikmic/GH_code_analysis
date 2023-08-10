# imports
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.gridspec as gridspec

def plot_children(fig, box, level=0, printit=True):
    '''
    Simple plotting to show where boxes are
    '''
    import matplotlib
    if isinstance(fig, matplotlib.figure.Figure):
        ax = fig.add_axes([0., 0., 1., 1.])
    else:
        ax = fig
    import matplotlib.patches as patches
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if printit:
        print("Level:", level)
    for child in box.children:
        rect = child.get_rect()
        if printit:
            print(child)
        ax.add_patch(
            patches.Rectangle(
                (child.left.value(), child.bottom.value()),   # (x,y)
                child.width.value(),          # width
                child.height.value(),          # height
                fc = 'none',
                ec = colors[level]
            )
        )
        if level%2 == 0:
            ax.text(child.left.value(), child.bottom.value(), child.name,
                   size=12-level, color=colors[level])
        else:
            ax.text(child.right.value(), child.top.value(), child.name, 
                    ha='right', va='top', size=12-level, color=colors[level])
        
        plot_children(ax, child, level=level+1, printit=printit)

fig, ax = plt.subplots(figsize=(3,3), tight_layout=False)
ax.plot(np.arange(0,50000,10000))
ax.set_ylabel("Ylabel"); ax.set_xlabel("Xlabel"); ax.set_title('AX0')
# fig.tight_layout()

fig = plt.figure(figsize=(5, 3))
gstop = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
gsleft = gridspec.GridSpecFromSubplotSpec(2, 2, gstop[0])
gsright = gridspec.GridSpecFromSubplotSpec(1, 1, gstop[1])
axsleft = []
for i in range(2):
    for j in range(2):
        axsleft += [fig.add_subplot(gsleft[i,j])]
axsleft[0].set_title('No Tight Layout')
axright = fig.add_subplot(gsright[0])  

fig = plt.figure(figsize=(5, 3))
gstop = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
gsleft = gridspec.GridSpecFromSubplotSpec(2, 2, gstop[0])
gsright = gridspec.GridSpecFromSubplotSpec(1, 1, gstop[1])
axsleft = []
for i in range(2):
    for j in range(2):
        axsleft += [fig.add_subplot(gsleft[i,j])]
axsleft[0].set_title('With Tight Layout')


axright = fig.add_subplot(gsright[0])
fig.tight_layout()

import LayoutBox as lb

figlb = lb.LayoutBox(parent=None, name='figlb')
figlb.set_geometry(0., 0., 1., 1.)

# make two boxes that will be laid out side by side
# they are constrained to be inside the parent.
leftlb = lb.LayoutBox(parent=figlb, name='leftlb')
rightlb = lb.LayoutBox(parent=figlb, name='rightlb')
# set their relation to one another
lb.hstack([leftlb, rightlb])
# set the right-side box width:
rightlb.set_width(0.25)

# this calls the solver
figlb.update_variables()

print(leftlb)
print(rightlb)

fig = plt.figure(figsize=(7,3))
# figure would have the below in it:
figlb = lb.LayoutBox(parent = None, name='figlb')
figlb.set_geometry(0., 0. ,1., 1.)

gs = gridspec.GridSpec(2,2)
# this would be called w/in GridSpec.  Note that it is not needed because 
# `gs` occupies all of fig, but that is not generally the case
gslb = lb.LayoutBox(parent=figlb, name='gslb')

axl = fig.add_subplot(gs[:,0])
axlsslb, axllb, axlspinelb = gslb.layout_axis_subplotspec(gs[:,0], ax=axl, name='l')
axl.set_visible(False)

axrt = fig.add_subplot(gs[0,1])
axrtsslb, axrtlb, axrtspinelb = gslb.layout_axis_subplotspec(gs[0,1], ax=axrt, name='rt')
axrt.set_visible(False)


axrb = fig.add_subplot(gs[1,1]); 
axrb.plot(np.arange(50000))
axrbsslb, axrblb, axrbspinelb = gslb.layout_axis_subplotspec(gs[1,1], ax=axrb, name='rb')
axrb.set_visible(False)

# here we make the margins match.
lb.match_margins([axlspinelb, axrtspinelb, axrbspinelb])
figlb.update_variables()
plot_children(fig, gslb, printit=False)

fig = plt.figure(figsize=(7,3))
# figure would have the below in it:
figlb = lb.LayoutBox(parent = None, name='figlb')
figlb.set_geometry(0., 0. ,1., 1.)

gs = gridspec.GridSpec(2,2)
# this would be called w/in GridSpec.  Note that it is not needed because 
# `gs` occupies all of fig, but that is not generally the case
gslb = lb.LayoutBox(parent=figlb, name='gslb')

axl = fig.add_subplot(gs[:,0])
axlsslb, axllb, axlspinelb = gslb.layout_axis_subplotspec(gs[:,0], ax=axl, name='l')
pcm=axl.pcolormesh(np.random.rand(32,32))
cbar = fig.colorbar(pcm, ax=axl)

# lets put a colorbar to the right of this one

cblb, clbspine = axllb.layout_axis_right(cbar.ax)

axl.set_visible(False)
cbar.ax.set_visible(False)


axrt = fig.add_subplot(gs[0,1])
axrtsslb, axrtlb, axrtspinelb = gslb.layout_axis_subplotspec(gs[0,1], ax=axrt, name='rt')
axrt.set_visible(False)


axrb = fig.add_subplot(gs[1,1]); 
axrb.plot(np.arange(50000))
axrbsslb, axrblb, axrbspinelb = gslb.layout_axis_subplotspec(gs[1,1], ax=axrb, name='rb')
axrb.set_visible(False)

# here we make the margins match.  We want to go two levels up because we want the 
# spines to match the subplotspec edges, *not* the parent axes...
lb.match_margins([axlspinelb, axrtspinelb, axrbspinelb], levels=2)
figlb.update_variables()
plot_children(fig, gslb, printit=False)

fig = plt.figure(figsize=(7,3))
# figure would have the below in it:
figlb = lb.LayoutBox(parent = None, name='figlb')
figlb.set_geometry(0., 0. ,1., 1.)

gs = gridspec.GridSpec(2,2)
# this would be called w/in GridSpec.  Note that it is not needed because 
# `gs` occupies all of fig, but that is not generally the case
gslb = lb.LayoutBox(parent=figlb, name='gslb')

axl = fig.add_subplot(gs[:,0])
axlsslb, axllb, axlspinelb = gslb.layout_axis_subplotspec(gs[:,0], ax=axl, name='l')
pcm=axl.pcolormesh(np.random.rand(32,32))

# lets put a colorbar to the right of this one




axrt = fig.add_subplot(gs[0,1])
axrtsslb, axrtlb, axrtspinelb = gslb.layout_axis_subplotspec(gs[0,1], ax=axrt, name='rt')
axrt.set_visible(False)


axrb = fig.add_subplot(gs[1,1]); 
axrb.plot(np.arange(50000))
axrbsslb, axrblb, axrbspinelb = gslb.layout_axis_subplotspec(gs[1,1], ax=axrb, name='rb')
axrb.set_visible(False)

cbar = fig.colorbar(pcm, ax=[axl, axrt, axrb])

cblb, clbspine = gslb.layout_axis_right(cbar.ax)



# here we make the margins match.
lb.match_margins([axlspinelb, axrtspinelb, axrbspinelb])


figlb.update_variables()
plot_children(fig, figlb, printit=False)



