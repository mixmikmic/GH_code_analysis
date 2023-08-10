
# problem_e.py


    
def check_collision(x, xr, xv, y, yv, yr):
    
    def distance(t):
        if len(x) != len(y):
            raise ValueError("Incompatible dimensions")
        xt = x + xv*t
        yt = y + yv*t
        dist = [(xi-yi)**2 for xi, yi in zip(xt, yt)]
        return math.sqrt(sum(dist))
    
    td = 0.001
    
    collision = False
    for i in range(10000000):
        if distance(td*i) < (xr+yr):
            collision = True
            print i
    
    if collision == False:
        print "No collision"
        
def tanner_method(x,y,xv,yv):
    deltaX = [xi-yi for xi,yi in zip(x,y)]
    deltaV = [xi-yi for xi,yi in zip(xv,yv)]
    
    print "deltaX:", deltaX
    print "deltaV:", deltaV
    
    t = (-1)*((deltaX[0]*deltaV[0] + deltaX[1]*deltaV[1] + deltaX[2]*deltaV[2])/(float(deltaV[0]**2 + deltaV[1]**2 + deltaV[2]**2)))
    return t
    
def time_of_first_contact(a,ar,av,b,br,bv):
    dx, dy, dz = [a[i] - b[i] for i in range(len(b))]
    dvx, dvy, dvz = [av[i] - bv[i] for i in range(len(b))]
    R = ar + br
    A = dvx*dvx + dvy*dvy + dvz*dvz
    B = 2*(dx*dvx + dy*dvy + dz*dvz)
    C = dx*dx + dy*dy + dz*dz - R*R
    
    
if __name__== "__main__":
    n = int(raw_input().strip())
    for i in xrange(n):
        params = [int(i) for i in raw_input().strip().split()]
        a = params[:3]
        ar = params[3]
        av = params[4:]
        params = [int(i) for i in raw_input().strip().split()]
        b = params[:3]
        br = params[3]
        bv = params[4:]
        
        print "av:", av
        print "bv:", bv
        
        t = tanner_method(a,b,av,bv)
        print 'time of closest approach:', t
        if t < 0:
            print "No collision"
        else:
            xt = x + t*xv
            yt = y = t*yv
            if xt:
                pass
            else:
                print "No collision"

# Problem E
import numpy as np

def ProbE():
    T = int(raw_input().strip())
    for test in xrange(T):
        craft = int(raw_input().strip().split(' '))
        junk = int(raw_input().strip().split(' '))
        x = craft[0]
        y = craft[1]
        z = craft[2]
        r1 = craft[3]
        xv = craft[4]
        yv = craft[5]
        zv = craft[6]
        _x = junk[0]
        _y = junk[1]
        _z = junk[2]
        r2 = junk[3]
        _xv = junk[4]
        _yv = junk[5]
        _zv = junk[6]

        a = (xv**2 - 2*xv*_xv + _xv**2 + yv**2 - 2*yv*_yv + _yv**2 + zv**2 - 2*zv*_zv + _zv**2)
        b = 2*(x*xv - _x*xv - x*_xv + _x*_xv + y*yv - _y*yv - y*_yv + _y*_yv + z*zv - _z*zv - z*_zv + _z*_zv)
        c = (x**2 - 2*x*_x + _x**2 + y**2 - 2*y*_y + _y**2 + z**2 - 2*z*_z + _z**2 - (r1+r2)**2)

        sol1 = (-b + np.sqrt(b**2 - 4*a*c))/2*a
        sol2 = (-b - np.sqrt(b**2 - 4*a*c))/2*a
        if b**2 < 4*a*c or (sol1 < 0 and sol2 < 0):
            print "No colllision"
        elif sol1 < 0:
            print(str(sol2))
        elif sol2 < 0:
            print(str(sol1))
        else:
            print(str(min(sol1,sol2)))

t = int(raw_input().strip())


for a0 in range(t):
    ob1 = map(float,raw_input().strip().split(" "))
    ob2 = map(float,raw_input().strip().split(" "))

    dp = [ob1[0]-ob2[0], ob1[1]-ob2[1], ob1[2]-ob2[2]]
    r = ob1[3]+ob2[3]
    dv = [ob1[4]-ob2[4], ob1[5]-ob2[5], ob1[6]-ob2[6]]

    #quadratic at^2+bt+c=0
    a = dv[0]**2 + dv[1]**2 + dv[2]**2
    b = 2*(dv[0]*dp[0] + dv[1]*dp[1] + dv[2]*dp[2])
    c = dp[0]**2 + dp[1]**2 + dp[2]**2-r**2

    #print dp,dv,r,a,b,c
    disc = b**2-4*a*c
    t_imp = -1
    if disc >= 0:
        t_imp = (-b-disc**0.5)/(2*a)

    if t_imp>0:
        print t_imp
    else:
        print "No collision"

def probE():
    t=int(raw_input())
    for i in xrange(t):
        collision=False
        x1,y1,z1,r1,vx1,vy1,vz1=raw_input().strip().split(" ")
        x2,y2,z2,r2,vx2,vy2,vz2=raw_input().strip().split(" ")
        x1,y1,z1,r1,vx1,vy1,vz1=int(x1),int(y1),int(z1),int(r1),int(vx1),int(vy1),int(vz1)
        x2,y2,z2,r2,vx2,vy2,vz2=int(x2),int(y2),int(z2),int(r2),int(vx2),int(vy2),int(vz2)
        x=np.linspace(0,100,10000)
        for time in x:
            loc1=(x1+vx1*time,y1+vy1*time,z1+vz1*time)
            loc2=(x2+vx2*time,y2+vy2*time,z2+vz2*time)
            loc1=np.array(loc1)
            loc2=np.array(loc2)
            distance=np.linalg.norm(loc1-loc2)
            if distance<=(r1+r2):
                print time
                collision=True
                break
        if collision==False:        
            print "No collision"

def problem_e():
    cases = int(raw_input())#how many cases
    for k in xrange(cases):
        spacecraft = raw_input().split()
        junk = raw_input().split()
        radius_craft = int(spacecraft[3])
        radius_junk = int(junk[3])
        total_radius = radius_junk + radius_craft
        space_start_x = int(spacecraft[0])
        space_start_y = int(spacecraft[1])
        space_start_z = int(spacecraft[2])
        junk_start_x = int(junk[0])
        junk_start_y = int(junk[1])
        junk_start_z = int(junk[2])
        space_dir_x = int(spacecraft[4])
        space_dir_y = int(spacecraft[5])
        space_dir_z = int(spacecraft[6])
        junk_dir_x = int(junk[4])
        junk_dir_y = int(junk[5])
        junk_dir_z = int(junk[6])
        domain = np.linspace(0,100,100000)
        distances = []
        count = 0
        for t in domain:
            position_craft = [space_start_x + space_dir_x *t, space_start_y+ space_dir_y*t,space_start_z + space_dir_z*t]
            position_junk = [junk_start_x + junk_dir_x*t, junk_start_y +junk_dir_y*t, junk_start_z+junk_dir_z*t]
            distance = np.sqrt((position_craft[0] - position_junk[0])**2 + (position_craft[1] - position_junk[1])**2 + (position_craft[2] - position_junk[2])**2)
            distances.append(distance)
            count += 1
            if distance < total_radius:
                #print distance
                print t
                break
            #print count
            #print distances
            if count > 5:
                #print count
                #print distances
                if distances[count-1] > distances[count - 2]:
                    print "No collision"
                    break

