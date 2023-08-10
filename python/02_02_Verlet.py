class particle2(object):
    
    def __init__(self, mass=1., x=0., y=0., vx=0., vy=0.):
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
       
    def euler(self, fx, fy, dt):
        self.vx = self.vx + fx*dt
        self.vy = self.vy + fy*dt
        self.x = self.x + self.vx*dt
        self.y = self.y + self.vy*dt
        
    def get_force(self):  # returns force per unit of mass (acceleration)
        GM=4*PI*PI # We use astronomical units
        r = math.sqrt(self.x*self.x+self.y*self.y)
        r3 = r * r * r
        fx = -GM*self.x/r3
        fy = -GM*self.y/r3
        return (fx,fy)
        
    def verlet(self, dt):
        (fx,fy) = self.get_force() # before I move to the new position
        self.x += self.vx*dt + 0.5*fx*dt*dt
        self.y += self.vy*dt + 0.5*fy*dt*dt
        self.vx += 0.5*fx*dt
        self.vy += 0.5*fy*dt
        (fx,fy) = self.get_force() # after I move to the new position
        self.vx += 0.5*fx*dt
        self.vy += 0.5*fy*dt




