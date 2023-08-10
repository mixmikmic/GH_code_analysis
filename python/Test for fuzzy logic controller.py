import numpy as np
from scipy.linalg import expm
import random

class move_gym():
    def __init__(self):
        self.scope = 2.0
        self.states = 6
        self.actions = 4
    def reset(self, s=[]):
        if np.array( s ).shape[0] == 0:
            self.obstacle_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            self.obstacle_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1

            self.move_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            self.move_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1  
            while ( abs(self.obstacle_x-self.move_x) < 1.05 ) & ( abs(self.obstacle_y-self.move_y) < 1.05 ):
                self.move_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
                self.move_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1      

            self.target_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            self.target_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            while (  ( abs(self.obstacle_x-self.target_x) < 1.05 ) & ( abs(self.obstacle_y-self.target_y) < 1.05 )  ) |             (  ( abs(self.move_x-self.target_x) < 1.05 ) & ( abs(self.move_y-self.target_y) < 1.05 )  ):
                self.target_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
                self.target_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1        
        else:
            self.obstacle_x=s[0]
            self.obstacle_y=s[1]
            self.target_x=s[2]
            self.target_y=s[3]
            self.move_x=s[4]
            self.move_y=s[5]

        state=np.array([ self.obstacle_x, self.obstacle_y, self.target_x, self.target_y, self.move_x, self.move_y ])
        return state #, self.state2img(state)
    def step(self, action):
        velocity = 0.2
        if action==0: # up down right left
            self.move_y+=velocity
        if action==1:
            self.move_y-=velocity
        if action==2:
            self.move_x+=velocity
        if action==3:
            self.move_x-=velocity

#         if self.move_x > (self.scope+1.0):
#             self.move_x-=velocity
#         if self.move_x < (-self.scope-1.0):
#             self.move_x+=velocity
#         if self.move_y > (self.scope+1.0):
#             self.move_y-=velocity
#         if self.move_y < (-self.scope-1.0):
#             self.move_y+=velocity
        
        reward = -0.1
        done = False
        info = "^_^"
        if (  ( abs(self.obstacle_x-self.move_x) < 1.05 ) & ( abs(self.obstacle_y-self.move_y) < 1.05 )  ): 
            reward = -1.0 
            done = True
            info = "collision"

        elif (  ( abs(self.target_x-self.move_x) < 1.05 ) & ( abs(self.target_y-self.move_y) < 1.05 )  ):
            reward = 1.0
            done = True
            info = "reach"

        state=np.array([ self.obstacle_x, self.obstacle_y, self.target_x, self.target_y, self.move_x, self.move_y ])

        return state, reward,done,info
    def state2img(self, state):
        img = np.zeros([84,84,3], dtype=np.uint8)
        ( img[ int(42-(self.obstacle_y+0.5)*10):int(42-(self.obstacle_y-0.5)*10), int((self.obstacle_x-0.5)*10+42):int((self.obstacle_x+0.5)*10+42),2 ] ).fill(255)
        ( img[ int(42-(self.target_y+0.5)*10):int(42-(self.target_y-0.5)*10), int((self.target_x-0.5)*10+42):int((self.target_x+0.5)*10+42),1 ] ).fill(255)
        ( img[ int(42-(self.move_y+0.5)*10):int(42-(self.move_y-0.5)*10), int((self.move_x-0.5)*10+42):int((self.move_x+0.5)*10+42),0 ] ).fill(255)
        return img

env = move_gym() # initialize env
STEP = 50 # Step limitation in an episode

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logic_q = np.load( '../logic_q-1.02-4-adjust-10.npy' )
discrete = np.array( [0., 1.02, 1.86, 2.93, 4.1, 95.03] )

def discrete_it(state, discrete):
    
    dis_s = []
    for i in xrange( len(state) ):
        num = state[i]
        over = num - discrete
        for j in xrange( len(over) ):
            if over[j]<0:
                over[j] = 100 # a certain big number
        dis_s.append( np.argmin( over ) )
    return dis_s

def state2relative(state):
    obstacle = np.array( state[0] )
    target = np.array( state[1] )
    agent = np.array( state[2][0] )
#     print agent
    
    top_t = 100.0
    bottom_t = 100.0
    right_t = 100.0
    left_t = 100.0

    top_o = 100.0
    bottom_o = 100.0
    right_o = 100.0
    left_o = 100.0        
#     print obstacle
#     print obstacle[:,1]
#     print abs(obstacle[:,1]-agent[1])
    obstacle_x=obstacle[np.argmin( abs(obstacle[:,0]-agent[0]) ),0]
    obstacle_y=obstacle[np.argmin( abs(obstacle[:,1]-agent[1]) ),1]
    target_x=target[np.argmin( abs(target[:,0]-agent[0]) ),0]
    target_y=target[np.argmin( abs(target[:,1]-agent[1]) ),1]
    move_x=agent[0]
    move_y=agent[1]

    if target_y >= move_y:
        top_t = target_y - move_y
    if target_y < move_y:
        bottom_t = move_y - target_y
    if target_x >= move_x:
        right_t = target_x - move_x
    if target_x < move_x:
        left_t = move_x - target_x

    if obstacle_y >= move_y:
        top_o = obstacle_y - move_y
    if obstacle_y < move_y:
        bottom_o = move_y - obstacle_y
    if obstacle_x >= move_x:
        right_o = obstacle_x - move_x
    if obstacle_x < move_x:
        left_o = move_x - obstacle_x

    return np.array([ top_t, bottom_t, right_t, left_t, top_o, bottom_o, right_o, left_o ])


state = env.reset(  )
obstacle_02=[state[0:2]]
A = [state]
old_action = 1.5
print state.tolist(),
for j in xrange(STEP):
    relative_state=state2relative( [ obstacle_02, [state[2:4]], [state[4:]] ] )
    # discrete it:
    relative_state = discrete_it( relative_state, discrete )
    print relative_state
    action = np.argmax( logic_q[relative_state[0]][relative_state[1]][relative_state[2]][relative_state[3]]    [relative_state[4]][relative_state[5]][relative_state[6]][relative_state[7]] )-1
    
    if (((relative_state[0]+relative_state[4])==0) | ((relative_state[1]+relative_state[5])==0)) & (action<1.2) & (old_action<1.2):
    #             print "^^^"
        action = old_action
    if (((relative_state[2]+relative_state[6])==0) | ((relative_state[3]+relative_state[7])==0)) & (action>1.8) & (old_action>1.8):
    #             print "%%%"
        action = old_action

    old_action = action
    
    state, reward,done,info = env.step(action)
    A.append(state)
    print '--', action, '->', state.tolist(),
    if done:
        print "info: ", info
        break

if True:
    i=0
    fig = plt.figure( figsize=(6,6) )
    ax = fig.add_subplot(111)

    ax.add_patch(
        patches.Rectangle(
            (-3, -3),
            6,
            6,
            color="grey"
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (-2, -2),
            4,
            4,
            color="black"
        )
    )

    s=np.array( A[i] )
    
    for i in range( len(obstacle_02) ):
        ax.add_patch(
            patches.Rectangle(
                (A[0][0]-0.5, A[0][1]-0.5),
                1,
                1,
                color="red",
                hatch="x"
            )
        )
    
#     ax.add_patch(
#         patches.Rectangle(
#             (s[0]-0.5, s[1]-0.5),
#             1,
#             1,
#             color="red",
#             hatch="x"
#         )
#     )

    ax.add_patch(
        patches.Rectangle(
            (A[0][2]-0.5, A[0][3]-0.5),
            1,
            1,
            color="green",
            hatch="+"
        )
    )

    ax.add_patch(
        patches.Rectangle(
            (A[0][4]-0.5, A[0][5]-0.5),
            1,
            1,
            color="blue",
            hatch="."
        )
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
#     plt.grid(True)
    
    for i in xrange( len(A)-1 ):
        ax.plot( [A[i][4], A[i+1][4]], [A[i][5], A[i+1][5]], color="white", linewidth=2)#plot lines
    
    plt.show()

    reward_keep=[0]

from tqdm import trange

B = []
B1 = []
for i in trange( 1000000 ):
    state = env.reset(  )
    obstacle_02=[state[0:2]]
    A = state
    old_action = 1.5
    for j in xrange(STEP):

        relative_state = state2relative( [ obstacle_02, [state[2:4]], [state[4:]] ] )
        # discrete it:
        relative_state = discrete_it( relative_state, discrete )

        action = np.argmax( logic_q[relative_state[0]][relative_state[1]][relative_state[2]][relative_state[3]]        [relative_state[4]][relative_state[5]][relative_state[6]][relative_state[7]] ) - 1
        
        
        if (((relative_state[0]+relative_state[4])==0) | ((relative_state[1]+relative_state[5])==0)) & (action<1.2) & (old_action<1.2):
#             print "^^^"
            action = old_action
        if (((relative_state[2]+relative_state[6])==0) | ((relative_state[3]+relative_state[7])==0)) & (action>1.8) & (old_action>1.8):
#             print "%%%"
            action = old_action
        
        old_action = action
        state, reward,done,info = env.step(action)
        if done:
            if info=="collision":
                B1.append(A)
            break
    if done == 0:
        B.append(A)

print "number of 'just wasting steps': ", len(B), "number of collision: ", len(B1)



