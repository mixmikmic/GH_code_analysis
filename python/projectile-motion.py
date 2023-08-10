get_ipython().magic('pylab inline')

class Projectile(object):

    # this is called every time a new object is created
    def __init__(self, v=1.0, theta=45, grav=9.81):

        self.v = v           # velocity m/s
        self.theta = theta   # angle (degrees)
        
        self.theta_rad = math.radians(theta)
        self.vx = v*math.cos(self.theta_rad)
        self.vy = v*math.sin(self.theta_rad)

        self.g = grav

        self.npts = 1000

    def height(self):

        # how high does this projectile go?
        # vf_y^2 = 0 = vi_y^2 - 2 g h
        h = self.vy**2/(2.0*self.g)

        return h

    def time(self):
        
        # time of flight up
        # vf_y = 0 = vi_y - g t
        t = self.vy/self.g

        # total time = up + down
        t = 2.0*t

        return t
        
    def distance(self):
        
        d = self.vx*self.time()

        return d

    def __str__(self):
        # a string representation for this class -- so we can print it
        str = " v: {} m/s\n theta: {} (degrees)\n height: {} m\n distance: {} m\n".format(
            self.v, self.theta, self.height(), self.distance())
            
        return str
    
    def t(self):
        return numpy.linspace(0.0, self.time(), num=self.npts)
    
    def x(self):
        return self.vx*self.t()
    
    def y(self):
        return self.vy*self.t() - 0.5*self.g*self.t()**2

p1 = Projectile(theta=80, v=10)
p2 = Projectile(theta=45, v=10)

print(p1)
print(p2)

pylab.plot(p1.x(), p1.y())
pylab.plot(p2.x(), p2.y())



