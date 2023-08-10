import numpy
from astropy.io import ascii
import matplotlib
from matplotlib import pyplot

get_ipython().run_line_magic('matplotlib', 'inline')

data = ascii.read('pie.csv', format='csv')
data

pyplot.axes().set_aspect('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
wedges, t1 = pyplot.pie(data['Percent'])#, shadow=True, radius=2)

pyplot.axes().set_aspect('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
wedges, t1 = pyplot.pie(data['Percent'], labels=data['Usage'])

pyplot.axes().set_aspect('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
wedges, t1, t2 = pyplot.pie(data['Percent'], labels=data['Usage'], autopct='%1.0f%%')

pyplot.axes().set_aspect('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
wedges, t1, t2 = pyplot.pie(data['Percent'], labels=data['Usage'], autopct='%1.0f%%')
pyplot.legend()

font = {'size' : 20}
matplotlib.rc('font', **font)

pyplot.axes().set_aspect('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
wedges, t1, t2 = pyplot.pie(data['Percent'], autopct='%1.0f%%', radius = 3)
pyplot.legend(wedges, data['Usage'], ncol=2, bbox_to_anchor = (-1., -1.3, 3, 0.5), mode='expand')

font = {'size' : 20}
matplotlib.rc('font', **font)

offset = 0.1
f = pyplot.figure(figsize=(4,4))
pyplot.axes().set_aspect('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
wedges, t1, t2 = pyplot.pie(data['Percent'], autopct='%1.0f%%', radius = 3, explode=(0,0,0,0,0, offset, offset), startangle = -30)
pyplot.legend(wedges, data['Usage'], ncol=2, bbox_to_anchor = (-1., -1.3, 3, 0.5), mode='expand')

#add a title
pyplot.title('Percentage of Poor Usage', position=(0.5,1.8), fontsize=30)
#define the group that we want to explode together, and find the angle it subtends
group = [0,1,2,3,4]
theta1 = min([wedges[i].theta1 for i in group])
theta2 = max([wedges[i].theta2 for i in group])
ang = (theta2 + theta1) / 2. * numpy.pi/180.
for i, (w,t,tn) in enumerate(zip(wedges, t1, t2)):
    #add lines around the wedges
    w.set_linewidth(2)
    w.set_edgecolor('#000000')
    tn.set_color('white')
    #this angle will be used for moving the percentage text
    wang = (w.theta2 + w.theta1) / 2. * numpy.pi/180.
    if (i in group):
        xoff = offset/2. * w.r * numpy.cos(ang)
        yoff = offset/2. * w.r * numpy.sin(ang)
        #move the wedge
        w.set_center( (w.center[0] + xoff, w.center[1] + yoff) )
        #move the text
        tn.set_position( ( 0.5*w.r * numpy.cos(wang) + xoff, 0.5*w.r * numpy.sin(wang) + yoff))#

#this is a silly fix for a lines bleeding out into the center of the figure in the PDF (not sure why python does this -- maybe it's dependent on OS or python version??)
x1 = offset*(0.75 + numpy.cos(theta1))
y1 = numpy.sin(theta1) - 0.935
y2 = numpy.sin(theta1) - 1
pyplot.plot([x1,x1],[y1,y1 + 0.1], color='white', linewidth=5)
pyplot.plot([x1,x1 + 0.1],[y1, y2], color='white', linewidth=5)


f.savefig('pie.pdf',format='pdf', bbox_inches = 'tight') 
f.savefig('pie.png',bbox_inches = 'tight') #png doesn't show these artifacts even without the silly fix



