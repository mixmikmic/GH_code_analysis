from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from IPython.display import display,HTML
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from matplotlib import patches
import seaborn

class VCRobot(object):
    '''
    Class to implement a velocity controlled two-wheeled robot
    
    The robot has state (x, y, theta) 
    The motion model is described in Probabilistic Robotics by Thrun ch. 5d
    '''
    def __init__(self, dt, alpha):
        '''
        member variables:
        dt - time step
        alpha - numpy array (4) describing noise characteristics
        '''
        self.dt = dt
        self.alpha = alpha
    
    def generate_motion(self,v,w,x0):
        '''
        inputs:
        v - numpy array (T) describing velocity
        w - numpy array (T) describing angular velocity
        x0 - numpy array (3) describing initial position
        
        outputs:
        X - numpy array (T,3) describing state at each time step
        '''
        dt = self.dt
        alpha = self.alpha
        T = len(v)
        
        X = np.zeros((T+1,3))
        X[0,:] = x0
        
        for t in xrange(T):
            var1 = alpha[0]*v[t]**2 + alpha[1]*w[t]**2
            var2 = alpha[2]*v[t]**2 + alpha[3]*w[t]**2
            v_hat = v[t] + np.sqrt(var1)*np.random.randn()
            w_hat = w[t] + np.sqrt(var2)*np.random.randn()
            stheta = np.sin(X[t,2])
            ctheta = np.cos(X[t,2])
            sthetap = np.sin(X[t,2] + dt*w_hat)
            cthetap = np.cos(X[t,2] + dt*w_hat)
            
            X[t+1,0] = X[t,0] - v_hat/w_hat*stheta + v_hat/w_hat*sthetap
            X[t+1,1] = X[t,1] + v_hat/w_hat*ctheta - v_hat/w_hat*cthetap
            X[t+1,2] = X[t,2] + w_hat*dt
        
        return X
        
class LocalizationAnimation(object):
    '''
    Class to visualize the 2D localization problem for a two-wheeled robot
    
    The robot has state (x, y, theta)
    The motion model is described in Probabilistic Robotics by Thrun ch. 5d
    '''
    def __init__(self, X, Y,dt):
        '''
        member variables:
        X - numpy array (T,3) state of the robot
        Y - numpy array (m,2) Locations of landmarks
        '''
        self.X = X
        self.Y = Y
        self.dt = dt
    
    def animate(self):
        '''
        X - numpy array (3,T) describing state at each time step
        '''
        X = self.X
        Y = self.Y
        dt = self.dt
        T = X.shape[0]
        
        xm = -20
        xM = 20
        ym = -20
        yM = 20
        
        env = plt.figure(figsize=(8, 8))
        ax = env.add_subplot(1, 1, 1)
        ax.set_xlim((xm, xM))
        ax.set_ylim((ym, yM))
        ax.set_title("Robot Environment",fontsize=20)
        ax.set_xlabel("X position (m)", fontsize=16)
        ax.set_ylabel("Y position (m)", fontsize=16)
        
        robot = plt.Circle((X[0, 0], X[0, 1]), 1, fill=False)
        ax.add_artist(robot)
        direction = np.array([[0, np.cos(X[0, 2])], [0, np.sin(X[0, 2])]])
        line, = ax.plot(X[0, 0] + direction[0, :], X[1, 0] + direction[1, :], 'k')
        features, = ax.plot(Y[:, 0], Y[:, 1], 'gD')

        def animate(i):
            direction = np.array([[0, np.cos(X[i, 2])], [0, np.sin(X[i, 2])]])
            line.set_data(X[i, 0] + direction[0, :], X[i, 1] + direction[1, :])
            robot.center = X[i, 0], X[i, 1]
            return (line,)
        
        anim = animation.FuncAnimation(env, animate, frames=T, interval=dt*500, blit=True)
        return anim

dt = .1
alpha = np.array([.1, .01, .01, .1])
t = np.arange(0,40.1, dt)
x0 = np.array([-5, -3, np.pi/2])
N_landmarks = 50
#Y = np.array([[6., 4., 0],
#              [-7., 8., 1],
#              [6., -4, 2]])
Y = np.zeros((N_landmarks,3))
Y[:,0] = np.random.uniform(low=-20., high=20., size=N_landmarks)
Y[:,1] = np.random.uniform(low=-20., high=20., size=N_landmarks)
Y[:,2] = np.arange(N_landmarks)
R = np.array([[.001, .0, .0],
              [.0, .001, .0],
              [.0, .0, .0001]])

v = 1 + .5*np.cos(.4*np.pi*t)
w = -.2 + 2*np.cos(1.2*np.pi*t)
robot = VCRobot(dt, alpha)
X = robot.generate_motion(v,w,x0)
ani = LocalizationAnimation(X,Y,dt)
HTML(ani.animate().to_html5_video())

class LandmarkSensor(object):
    '''
    Class to simulate a landmark sensor
    This is a basic range-bearing sensor in 2D
    '''
    
    def __init__(self, Y, R, alpha):
        '''
        member variables:
        Y - numpy array (m, 2) locations of the landmarks
        R - noise covariance associated with range and bearing
        alpha - FOV of the camera in radians
        '''
        self.Y = Y
        self.R = R
        self.alpha = alpha
        
    def simulate(self, X):
        '''
        inputs:
        X - numpy array (T,3) of vehicle state (x,y, theta)
        
        outputs:
        Z - numpy array (T,3*m) of range/bearing/signature measurements
        '''
        Y = self.Y
        m, _ = Y.shape
        
        # Note: this will need to be fixed when we limit the FOV of the robot
        Z = []
        
        for i in xrange(m):
            z = np.zeros(3)
            z[0] = np.linalg.norm(X[:2] - self.Y[i,:2])
            z[1] = np.arctan2(self.Y[i,1] - X[1], self.Y[i,0] - X[0]) - X[2]
            
            z += np.random.multivariate_normal(np.zeros(3), self.R)
            # wrap relative bearing
            if z[1] > np.pi:
                z[1] = z[1] - 2*np.pi
            if z[1] < -np.pi:
                z[1] = z[1] + 2*np.pi
            z[2] = Y[i,2]
            if np.abs(z[1]) < self.alpha/2:
                Z.append(z)
        
        Z = np.array(Z) 
        return Z

rbsensor = LandmarkSensor(Y, R, np.pi/4)
Z = rbsensor.simulate(X[:,1:2])

class EKFSLAM(object):
    '''
    Class to implement an Extended Kalman Filter SLAM
    for a system with noise on control inputs
    '''
    
    def __init__(self,x0, f,h,F,G,H,Q,R):
        '''
        Class Variables:
        f(x,u) - function handle nonlinear state transition
        h(x,u) - function handle nonlinear measurement model
        F(x,u) - function handle Jacobian of state w/ respect to state
        G(x,u) - function handle Jacobian of state w/ respect to input
        H(x,u) - function handle Jacobian of measurement w/ respect to state
        Q(u) - function handle Covariance of input
        R - numpy matrix (mxm) sensor noise covariance
        '''
        inf = 1e15
        self.mu = np.zeros(3*N_landmarks + 3)
        self.mu[0:3] = x0
        self.cov = np.matrix(inf*np.eye(3*N_landmarks + 3))
        self.cov[0,0] = 0
        self.cov[1,1] = 0
        self.cov[2,2] = 0
        self.seen = N_landmarks*[False]
        self.f = f
        self.h = h
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
    
    def filter(self,z,u):
        '''
        z - (n_landmarks_seen x 3)
        u - control input
        Filters the the slam problem over 1 time step
        '''
        # Prediction
        self.mu[0:3] = self.f(self.mu[0:3],u)
        Fx = np.matrix(np.zeros((3,3*N_landmarks + 3)))
        Fx[0,0] = 1
        Fx[1,1] = 1
        Fx[2,2] = 1
        F = np.matrix(np.eye(3*N_landmarks+3))+Fx.T*self.F(self.mu[0:3],u)*Fx
        G = 5*self.G(self.mu[0:3],u)
        self.cov = F*self.cov*F.T + Fx.T*G*self.Q(u)*G.T*Fx
        if z.size == 0:
            return np.copy(self.mu), np.copy(self.cov)
        
        l = z[:,2]
        for j,s in enumerate(l):
            #Measurement Update
            s = int(s)
            if self.seen[s] == False:
                self.mu[3*(s+1)] = self.mu[0] + z[j, 0]*np.cos(z[j,1] + self.mu[2])
                self.mu[3*(s+1) + 1] = self.mu[1] + z[j, 0]*np.sin(z[j,1] + self.mu[2])
                self.mu[3*(s+1) + 2] = s
                self.seen[s] = True
                
            y = self.mu[3*(s+1):3*(s+1)+3]
            z_hat = self.h(self.mu[:3],y,u)
            Fxj = np.matrix(np.zeros((6, 3*N_landmarks + 3)))
            Fxj[0,0] = 1
            Fxj[1,1] = 1
            Fxj[2,2] = 1
            Fxj[3,3*(s+1)] = 1
            Fxj[4, 3*(s+1)+1] = 1
            Fxj[5, 3*(s+1)+2] = 1
            H = self.H(self.mu[:3],y,u)*Fxj
            K = self.cov*H.T*(la.inv(H*self.cov*H.T + self.R))
            innovation = z[j] - z_hat
            if innovation[1] > np.pi:
                innovation[1] -= 2*np.pi
            elif innovation[1] < -np.pi:
                innovation += 2*np.pi
            innovation = np.matrix(innovation).T
            update = np.array(K*innovation).flatten()
            self.mu += update
            self.cov = (np.eye(3*N_landmarks+3) - K*H).dot(self.cov)
            
        return np.copy(self.mu), np.copy(self.cov)

def f(x,u):
    xp = np.zeros_like(x)
    v = u[0]
    w = u[1]
    stheta = np.sin(x[2])
    ctheta = np.cos(x[2])
    sthetap = np.sin(x[2] + dt*w)
    cthetap = np.cos(x[2] + dt*w)

    xp[0] = x[0] - v/w*stheta + v/w*sthetap
    xp[1] = x[1] + v/w*ctheta - v/w*cthetap
    xp[2] = x[2] + w*dt
    return xp

def h(x,y,u):
    zp = np.zeros(3)
    
    zp[0] = np.linalg.norm(x[0:2] - y[0:2])
    zp[1] = np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2]
    if zp[1] > np.pi:
        zp[1] -= 2*np.pi
    elif zp[1] < -np.pi:
        zp[1] += 2*np.pi
    zp[2] = y[2]
    return zp

def F(x,u):
    n = x.shape[0]
    v = u[0]
    w = u[1]
    stheta = np.sin(x[2])
    ctheta = np.cos(x[2])
    sthetap = np.sin(x[2] + dt*w)
    cthetap = np.cos(x[2] + dt*w)
    
    F = np.matrix(np.zeros((n,n)))
    F[0,2] = -v/w*ctheta + v/w*cthetap
    F[1,2] = -v/w*stheta + v/w*sthetap
    return F
    
def G(x,u):
    n = x.shape[0]
    k = u.shape[0]
    v = u[0]
    w = u[1]
    stheta = np.sin(x[2])
    ctheta = np.cos(x[2])
    sthetap = np.sin(x[2] + dt*w)
    cthetap = np.cos(x[2] + dt*w)
    
    G = np.matrix(np.zeros((n,k)))
    G[0,0] = (-stheta + sthetap)/w
    G[0,1] = v*(stheta-sthetap)/(w**2) + v*(ctheta*dt)/w
    G[1,0] = (ctheta - cthetap)/w
    G[1,1] = -v*(ctheta - cthetap)/(w**2) + v*(stheta*dt)/w
    G[2,1] = dt
    return G

def H(x,y,u):
    H = np.matrix(np.zeros((3,6)))
    dx = y[0] - x[0]
    dy = y[1] - x[1]
    q = (y[0] - x[0])**2 + (y[1] - x[1])**2
    sq = np.sqrt(q)
    H[0,0] = -(dx)/sq
    H[0,1] = -(dy)/sq
    H[0,2] = 0
    H[0,3] = dx/sq
    H[0,4] = dy/sq
    H[0,5] = 0
    H[1,0] = dy/q
    H[1,1] = -dx/q
    H[1,2] = -1
    H[1,3] = -dy/q
    H[1,4] = dx/q
    H[1,5] = 0
    H[2,5] = 1
    return H

def Qt(u):
    k = u.shape[0]
    v = u[0]
    w = u[1]
    Q = np.matrix(np.zeros((k,k)))
    Q[0,0] = alpha[0]*v**2 + alpha[1]*w**2
    Q[1,1] = alpha[2]*v**2 + alpha[3]*w**2
    return Q

U = np.column_stack([v, w])
x0 = np.array([-5, -3, np.pi/2])
x_hat = x0

ekf = EKFSLAM(x0,f,h,F,G,H,Qt,R)

X_hat = []
Cov_hat = []
for t,u in enumerate(U):
    z = rbsensor.simulate(X[t+1])
    x_hat, Cov = ekf.filter(z,u)
    X_hat.append(x_hat)
    Cov_hat.append(Cov)
    

class SLAMAnimation(object):
    '''
    Class to visualize the SLAM problem for a two-wheeled robot
    
    The robot has state (x, y, theta)
    The motion model is described in Probabilistic Robotics by Thrun ch. 5d
    '''
    def __init__(self, X, L, Cov, dt):
        '''
        member variables:
        X - list (T,3*Num_landmarks + 3) state of the robot and landmarks
        Y - numpy array (m,2) Locations of landmarks
        '''
        self.X = X
        self.L = Y
        self.Cov = Cov
        self.dt = dt
    
    def animate(self):
        '''
        X - numpy array (3,T) describing state at each time step
        '''
        X = self.X
        Cov = self.Cov
        dt = self.dt
        T = len(X)
        
        xm = -20
        xM = 20
        ym = -20
        yM = 20
        
        env = plt.figure(figsize=(8, 8))
        ax = env.add_subplot(1, 1, 1)
        ax.set_xlim((xm, xM))
        ax.set_ylim((ym, yM))
        ax.set_title("Robot Environment",fontsize=20)
        ax.set_xlabel("X position (m)", fontsize=16)
        ax.set_ylabel("Y position (m)", fontsize=16)
        
        robot = plt.Circle((X[0][0], X[0][1]), 1, fill=False)
        ax.add_artist(robot)
        direction = np.array([[0, np.cos(X[0][2])], [0, np.sin(X[0][2])]])
        line, = ax.plot(X[0][0] + direction[0, :], X[0][1] + direction[1, :], 'k')
        land, = ax.plot(self.L[:,0], self.L[:,1], 'gD',markersize=3)
        Y = X[0][3:].reshape(N_landmarks, 3)
        features, = ax.plot(Y[:, 0], Y[:, 1], 'rD', markersize=3)
        ells = []
        for _ in xrange(N_landmarks):
            ell = patches.Ellipse((0.,0.), 100., 100., color='b', linewidth=1)
            ell.set_facecolor('none')
            ax.add_artist(ell)
            ells.append(ell)

        def animate(i):
            direction = np.array([[0, np.cos(X[i][2])], [0, np.sin(X[i][2])]])
            line.set_data(X[i][0] + direction[0, :], X[i][1] + direction[1, :])
            robot.center = X[i][0], X[i][1]
            Y = X[i][3:].reshape(N_landmarks, 3)
            features.set_data(Y[:, 0], Y[:, 1])
            C = Cov[i]
            for j, ell in enumerate(ells):
                Cj = C[3*(j+1):3*(j+1)+2,3*(j+1):3*(j+1)+2]
                lamb, v = np.linalg.eig(Cj)
                if lamb[0] > 100:
                    continue
                ell.center = (Y[j,0],Y[j,1])
                ell.height = lamb[1]*4
                ell.width = lamb[0]*4
                ell.angle = np.rad2deg(np.arccos(v[0,0]))
            return (line,features)
        
        anim = animation.FuncAnimation(env, animate, frames=T, interval=dt*500, blit=True)
        return anim
    
ani = SLAMAnimation(X_hat, Y, Cov_hat,dt)
HTML(ani.animate().to_html5_video())

X_hat = np.array(X_hat)

plt.plot(X[:,0],X[:,1],label="Truth")
plt.plot(X_hat[:,0], X_hat[:,1],label="Estimate")
plt.title("Robot Location")
plt.legend()
plt.show()

plt.plot(X[:,0],label="Truth")
plt.plot(X_hat[:,0],label="Estimate")
plt.title("X Position")
plt.legend()
plt.show()

plt.plot(X[:,1],label="Truth")
plt.plot(X_hat[:,1],label="Estimate")
plt.title("Y Position")
plt.legend()
plt.show()

plt.plot(X[:,2],label="Truth")
plt.plot(X_hat[:,2],label="Estimate")
plt.title("Theta")
plt.legend()
plt.show()

