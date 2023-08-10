get_ipython().run_line_magic('run', "'Set-up.ipynb'")

from pyrep import VRep
from pyrep.vrep import vrep as vrep

def loadSceneRelativeToClient(path='../scenes/Pioneer.ttt',
                              ip='127.0.0.1',
                              port=19997,
                              waitUntilConnected=True,
                              doNotReconnectOnceDisconnected=True,
                              timeOutInMs=5000,
                              commThreadCycleInMs=5):
    vrep.simxFinish(-1)
    clientID=vrep.simxStart(ip,port,waitUntilConnected,doNotReconnectOnceDisconnected,timeOutInMs,commThreadCycleInMs)
    #Works - relative to V-REP executable location, absolute path
    #vrep.simxLoadScene(clientID,'/Applications/V-REP_PRO_EDU_V3_4_0_Mac/scenes/collisionDetectionDemo.ttt',0x00,vrep.simx_opmode_blocking)
    #Works - relative to remote API client location, absolute path 
    #vrep.simxLoadScene(clientID,'/Users/ajh59/Pioneer.ttt',0xFF,vrep.simx_opmode_blocking)
    #Works relative to remote API client location, relative path
    vrep.simxLoadScene(clientID,path,0xFF,vrep.simx_opmode_blocking)
    vrep.simxFinish(-1)

