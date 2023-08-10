#NAME: Coupled Points
#DESCRIPTION: Interactive 3D simulation of systems of particles and springs.

from math import sqrt
from vpython import *
from __future__ import division, print_function
from ipywidgets import widgets

#should not be changed
points=[]
springs=[]
GRAVITY=False
COLLISIONS=False
flag=0 #start 1, end -1, or 0 if not attached to spring

# can be changed
DELTAt=0.001
tMAX=100.0
R=0.4
g=-9.81
HELICES=True #whether springs should be visible or not

def sub(e1,e2):
    return [e1[0]-e2[0],e1[1]-e2[1],e1[2]-e2[2]]

def add(e1,e2):
    return [e1[0]+e2[0],e1[1]+e2[1],e1[2]+e2[2]]

def sMul(a,e1):
    return ([a*e1[0],a*e1[1],a*e1[2]])

def dot(e1,e2):
    return (e1[0]*e2[0]+e1[1]*e2[1]+e1[2]*e2[2])

def NORM(e1):
    return sqrt(e1[0]**2+e1[1]**2+e1[2]**2)

def unit(e1):
    try:
        return sMul(1.0/NORM(e1),e1)
    except ZeroDivisionError:
        return [0.0,0.0,0.0]

def conVec(e1):
    vec=vector(0.0,0.0,0.0)
    vec.x=e1[0]
    vec.y=e1[1]
    vec.z=e1[2]
    return vec
    
def fl(string,m=0,r=0):
    try:
        return float(string)
    except ValueError:
        if(m):
            return 1.0
        else:
            if(r):
                return R
            else:
                return 0.0

def INT(string):
    try:
        return int(string)
    except ValueError:
        return 0

def allowedRange(a):
    if(a>=0 and a<len(points)):
        return True
    else:
        return False

class point:
    def __init__(self,r0,v0=[0.0,0.0,0.0],m0=1.0,fixed=0,RADIUS=R):
        self.r=r0
        self.v=v0
        self.radius=RADIUS
        if m0==0.0:
            self.m=0.0001
        else:
            self.m=m0
        self.F=[0.0,0.0,0.0]
        self.FIXED=fixed
        if(self.FIXED):
            self.v=[0.0,0.0,0.0]
    def vUpdate(self):
        self.v=add(self.v,sMul(DELTAt/self.m,self.F))
    def rUpdate(self):
        self.r = add(self.r,add(sMul(DELTAt,self.v),sMul(0.5*(DELTAt**2)/self.m,self.F)))

class spring:   
    def length(self):
        return NORM(sub(points[self.start].r,points[self.end].r))
    def force(self):
        return self.k*(self.L-self.L0)
    def __init__(self,point1,point2,l0,K):
        self.start=point1
        self.end=point2
        self.L0=l0
        self.L=self.length()
        self.k=K
        self.F=self.force()
        self.rs=unit(sub(points[self.end].r,points[self.start].r))
        self.re=unit(sub(points[self.start].r,points[self.end].r))
    def update(self):
        self.L=self.length()
        self.F=self.force()
        self.rs=unit(sub(points[self.end].r,points[self.start].r))
        self.re=unit(sub(points[self.start].r,points[self.end].r))

def impulse(i,j):
    if(i==j):
        pass
    #else:
    #    n=unit(sub(points[i].r,points[j].r))
    #    component=sMul(2.0*dot(n,sub(points[i].v,points[j].v))/(points[i].m*((1.0/points[i].m)+(1.0/points[j].m))),n)
    #    points[i].v=add(points[i].v,component)
    #    points[i].r=add(points[i].r,sMul(5.0*DELTAt,points[i].v))
    
def collisions(i):
    for j in range(0,len(points)):
            if(NORM(sub(points[i].r,points[j].r))<=(points[i].radius+points[j].radius)):
                impulse(i,j)
                break

def updatePoints():
    for i in range(0,len(points)):
        points[i].F=[0.0,0.0,0.0]
        for j in range(0,len(springs)):
            if(springs[j].start==i):
                flag=1
            elif(springs[j].end==i):
                flag=-1
            else:
                flag=0
            if(points[i].FIXED):
                points[i].F=[0.0,0.0,0.0]
            else:
                if(flag==1):
                    points[i].F=add(points[i].F,sMul(springs[j].F,springs[j].rs))
                elif(flag==-1):
                    points[i].F=add(points[i].F,sMul(springs[j].F,springs[j].re))
                else:
                    pass
        if (GRAVITY and (not points[i].FIXED)):
            points[i].F=add(points[i].F,sMul(g*points[i].m,[0,1,0]))
        if(COLLISIONS):
            collisions(i)
        points[i].vUpdate()
        points[i].rUpdate()
    
    for i in range(0,len(springs)):
        springs[i].update()

class vis3D:
    def __init__(self):
        self.BALL=[]
        self.scene = canvas()
        for i in range(0,len(points)):
            self.BALL.append(sphere(pos=conVec(points[i].r), radius=points[i].radius))
            if(points[i].FIXED):
                 self.BALL[i].color=color.red
            else:
                self.BALL[i].color=color.green
        if HELICES:
            self.HELIX=[]
            for i in range(0,len(springs)):
                self.HELIX.append(helix(pos=conVec(points[springs[i].start].r),axis=conVec(sMul(springs[i].L,springs[i].rs)), radius=0.4, color=color.blue))
    def update(self):
        updatePoints()
        for i in range(0,len(points)):
            self.BALL[i].pos=conVec(points[i].r)
        if HELICES:
            for i in range(0,len(springs)):
                self.HELIX[i].pos=conVec(points[springs[i].start].r)
                self.HELIX[i].axis=conVec(sMul(springs[i].L,springs[i].rs))
    def run(self):
        t=0.0
        self.scene.background=color.white
        display(self.scene)
        while t<tMAX:
            rate(int(1.0/DELTAt))
            self.update()
            t+=DELTAt

class gui:
    def __init__(self):
        self.pos = [widgets.Text(description='X',width=100),widgets.Text(description='Y',width=100 ),widgets.Text(description='Z',width=100)]
        self.POS = widgets.HBox(children=self.pos)
        self.vel=[widgets.Text(description='Vx',width=100),widgets.Text(description='Vy',width=100 ),widgets.Text(description='Vz',width=100)]
        self.VEL=widgets.HBox(children=self.vel)
        self.misc=[widgets.Text(description='Mass',width=100),widgets.Text(description='Radius',width=100),widgets.widget_bool.Checkbox(description='Fixed',width=100)]
        self.MISC=widgets.HBox(children=self.misc)
        self.create=widgets.Button(description="Create Point",width=100)
        self.NEXT = widgets.Button(description="Next",width=100)
        self.sprAtt = [widgets.Text(description='Start',width=100),widgets.Text(description='End',width=100 )]
        self.SPRATT = widgets.HBox(children=self.sprAtt)
        self.sprProp = [widgets.Text(description='L0',width=100),widgets.Text(description='K',width=100 )]
        self.SPRPROP = widgets.HBox(children=self.sprProp)
        self.createSpr=widgets.Button(description="Create Spring",width=100)
        self.grav=[widgets.widget_bool.Checkbox(description='Gravity',width=100),widgets.widget_bool.Checkbox(description='Collisions',width=100)]
        self.GRAV=widgets.HBox(children=self.grav)
        self.START=widgets.Button(description="Start",width=100)
        self.create.on_click(self.addPoint)
        self.NEXT.on_click(self.nxt)
        self.createSpr.on_click(self.addSpring)
        self.START.on_click(self.start)

    def display(self):
        display(self.POS,self.VEL,self.MISC,self.create,self.NEXT)

    def addPoint(self,b):
        points.append(point([fl(self.pos[0].value),fl(self.pos[1].value),fl(self.pos[2].value)],[fl(self.vel[0].value),fl(self.vel[1].value),fl(self.vel[2].value)],fl(self.misc[0].value,m=1),INT(self.misc[2].value),fl(self.misc[1].value,r=1)))
        if(points[len(points)-1].FIXED):
            print("Fixed Particle " +str(int(len(points)))+" Created. r0="+str(points[len(points)-1].r)+"m. v0="+str(points[len(points)-1].v)+"m/s. mass="+str(points[len(points)-1].m)+"kg. radius="+str(points[len(points)-1].radius)+"m.")
        else:
            print("Movable Particle " +str(int(len(points)))+" Created. r0="+str(points[len(points)-1].r)+"m. v0="+str(points[len(points)-1].v)+"m/s. mass="+str(points[len(points)-1].m)+"kg. radius="+str(points[len(points)-1].radius)+"m.")
    
    def nxt(self,b):
        display(self.SPRATT,self.SPRPROP,self.GRAV,self.createSpr,self.START)
        #make plot of point location numbered
        
    def addSpring(self,b):
        if(not(allowedRange(INT(self.sprAtt[0].value)-1) and allowedRange(INT(self.sprAtt[1].value)-1))):
            print("Couldn't Create Spring")
        else:
            springs.append(spring(INT(self.sprAtt[0].value)-1,INT(self.sprAtt[1].value)-1,fl(self.sprProp[0].value,m=1),fl(self.sprProp[1].value,m=1)))
            print("Spring Created Between Particles " +str(INT(self.sprAtt[0].value))+" and " + str(INT(self.sprAtt[1].value))+". L0="+str(springs[len(springs)-1].L0)+"m. K="+str(springs[len(springs)-1].k)+"Nm.")
    def start(self,b):
        if self.grav[0].value:
            global GRAVITY
            GRAVITY = True
        if self.grav[1].value:
            global COLLISIONS
            COLLISIONS = False#True
        self.visual=vis3D()
        self.visual.run()

GUI=gui()
GUI.display()

