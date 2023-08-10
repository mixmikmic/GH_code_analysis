from ipywidgets import interact, interact_manual
import ipywidgets

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import numpy as np

import math

get_ipython().run_line_magic('run', "'Set-up.ipynb'")
get_ipython().run_line_magic('run', "'Loading scenes.ipynb'")

loadSceneRelativeToClient('../scenes/Baxter_demo.ttt')

from pyrep.vrep.vrep import simxGetObjectOrientation, simxGetObjectHandle, simxGetFloatSignal

rclass='Baxter_base'
print('Loading class: {}'.format(rclass))
class Baxter_base:

    def __init__(self, api: VRep):
        self._api = api
        self._joint1 = api.joint.with_position_control("Baxter_leftArm_joint4")
        #self._sensor_ultrasonic_left = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor3")
        #self._sensor_ultrasonic_right = api.sensor.proximity("Pioneer_p3dx_ultrasonicSensor6")
       
        res, self._handle = simxGetObjectHandle(self.id, 'Baxter', vrep.simx_opmode_oneshot_wait)
        
        self.joints= self._joints()
        self.sensors= self._joints()
        self.handles = self._introspect()
        self.names_by_handles = {self.handles[k]:k for k in self.handles}

    def _get_handle(self,name):
        res, handle=vrep.simxGetObjectHandle(self.id,
                                             name,
                                             vrep.simx_opmode_blocking)
        return handle
    
    def _introspect(self):
        #http://galvanicloop.com/blog/post/7/quadruped-robot-5-simulation-on-v-rep
        errorCode, handles, intData,         floatData, array = vrep.simxGetObjectGroupData(self.id,
                                                       vrep.sim_appobj_object_type,
                                                       0,
                                                       vrep.simx_opmode_oneshot_wait)
        return dict(zip(array, handles))
    
    def _joints(self):
        j = self._introspect()
        #Add arm joints
        joints={k: j[k] for k in j if 'joint' in k}
        #Add monitor joint
        joints['Baxter_monitorJoint']=j['Baxter_monitorJoint']
        return joints
    
    def _sensors(self):
        s = self._introspect()
        return {k: j[k] for k in s if 'ensor' in k}
    
    def get_joint_angle(self,jointname, degrees=False):
        handle=self.joints[jointname]
        res, pos = vrep.simxGetJointPosition(self.id,
                                                 handle,
                                                 vrep.simx_opmode_blocking)
        if degrees:
            pos = pos * 180 / math.pi
        return pos
        
    def joint_angles(self, degrees=False):
        ja = {}
        for j in sorted(self.joints):
            pos = self.get_joint_angle(j,degrees=degrees)
            ja[j] = pos
        return ja
    
'''
    def get_orientation(self):
        #http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctions.htm#simxGetObjectOrientation
        #Returns a value between +/-pi
        return simxGetObjectOrientation(self.id, self._handle, -1, v.simx_opmode_streaming)[1]
'''
        
print('This is a base class for the {} model\n'.format(eval(rclass).__name__ ))

rclass='Baxter'
print('Loading class: {}'.format(rclass))
class Baxter(Baxter_base):

    def __init__(self, api: VRep):
        self._api = api
        self.id = api._id
        
        tmp1,tmp2=self.get_coords_left_tip(True),self.get_coords_right_tip(True)
        
        #Inherit init settings from parent class
        super(Baxter, self).__init__(api)

    def set_joint_angle(self, joint_name, angle):
        ''' Set the joint angle of a joint referred to by joint name '''
        #The joint angle is set by reference to the joint handle
        #Look-up the joint handle from the joint name
        handle=self.joints[joint_name]
        #res,handle = vrep.simxGetObjectHandle(self.id,'Baxter_rightArm_joint4',vrep.simx_opmode_oneshot_wait); 
        #Set the joint angle
        vrep.simxSetJointTargetPosition(self.id,
                                        handle,
                                        angle,
                                        vrep.simx_opmode_oneshot);
        
    def _get_coords_tip(self,arm,init=False):
        #simx_opmode_streaming (the first call) thence simx_opmode_buffer
        handle=self._get_handle('Baxter_{}Arm_tip'.format(arm))
        
        if init: mode= vrep.simx_opmode_buffer
        else: mode =vrep.simx_opmode_streaming
            
        res,pos=vrep.simxGetObjectPosition(self.id,handle,
                                           -1, mode)
        return pos

    def get_coords_left_tip(self, init=False):
        return self._get_coords_tip('left', init)
    
    def get_coords_right_tip(self,init=False):
        return self._get_coords_tip('right', init)
    
    def get_vision_sensor_image(self, vision_sensor_name):
        #http://www.forum.coppeliarobotics.com/viewtopic.php?f=9&t=7012&p=27786

        res, v1 = vrep.simxGetObjectHandle(self.id, vision_sensor_name, vrep.simx_opmode_oneshot_wait)
        err, resolution, image = vrep.simxGetVisionSensorImage(self.id, v1, 0, vrep.simx_opmode_streaming)
        img=None
        while err!=vrep.simx_return_ok:#(vrep.simxGetConnectionId(clientID) != -1):
            err, resolution, image = vrep.simxGetVisionSensorImage(self.id, v1, 0, vrep.simx_opmode_buffer)
            if err == vrep.simx_return_ok:
                #print("image OK!!!")
                img = np.array(image,dtype=np.uint8)
                #
                img.resize([resolution[1],resolution[0],3])
                #For some reason the image is upside down unless we flip it?
                img = cv2.flip(img,0)
                ok=False
            elif err == vrep.simx_return_novalue_flag:
                #print("no image yet")
                pass
            else:
                print(err)
        return img
    
    def get_ultrasonic_sensor_reading(self, ultrasonic_sensor_number):
        if '{}'.format(ultrasonic_sensor_number).isdigit() and int(ultrasonic_sensor_number) > 0 and int(ultrasonic_sensor_number)<13: 
            ultrasonic_sensor_number=int(ultrasonic_sensor_number)
        else: return "Not a valid input: expecting int in range 1..12"
        handle = self._get_handle('Baxter_ultrasonic_sensor{}'.format(ultrasonic_sensor_number))
        err, detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.id,handle,vrep.simx_opmode_streaming)
        while err!=vrep.simx_return_ok:
            err, detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.id,handle,vrep.simx_opmode_buffer)
        if not detectionState:
            return False
        distance=math.sqrt(detectedPoint[0]*detectedPoint[0]+detectedPoint[1]*detectedPoint[1]+detectedPoint[2]*detectedPoint[2])
        return distance, self.names_by_handles[detectedObjectHandle], detectedPoint, detectedSurfaceNormalVector

methods = [method for method in dir(eval(rclass)) if not method.startswith('_')]
print('Methods available in {}:\n\t{}\n'.format(eval(rclass).__name__ , '\n\t'.join(methods)))        

from pyrep import VRep
from pyrep.vrep import vrep as vrep

#Ensure there are no outstanding simulations running
vrep.simxFinish(-1)

#Open  connection to the simulator
api=VRep.connect("127.0.0.1", 19997)

#Start the simulation
api.simulation.start()

#Create a Python object to represent the simulated robot
r = Baxter(api)

plt.imshow( r.get_vision_sensor_image('Baxter_rightArm_camera') );

plt.imshow( r.get_vision_sensor_image('Baxter_leftArm_camera') );

r.joint_angles()

r.joint_angles(degrees=True)

def f(j, x):
    r.set_joint_angle(j,x)
    
interact_manual(f, j=['Baxter_leftArm_joint1',
                      'Baxter_leftArm_joint2',
                      'Baxter_monitorJoint'],
                x=(-2,2,0.2));

posDisplay = ipywidgets.Text()



for j in r.joints:
    exec("""
def {j}({js}):
    r.set_joint_angle('{j}',{js})
    posDisplay.value=','.join([str(x) for x in r.get_coords_left_tip()])+ ','.join([str(x) for x in r.get_coords_right_tip()])
interact({j}, {js}=(-3.5,3.5,0.2))
""".format(j=j, js='_'.join(j.split('_')[1:]).replace('Arm_joint','')))
posDisplay

','.join([str(x) for x in r.get_coords_left_tip()])+ ','.join([str(x) for x in r.get_coords_right_tip()])

r.get_coords_left_tip(), r.get_coords_right_tip()

joints_range={}
for j in r.joints:
    r.set_joint_angle(j,0)
    
for j in r.joints:
    joint_min=999
    joint_max=-999
    joint_curr=0
    print('Looking for max {}...'.format(j))
    r.set_joint_angle(j,0)
    if j.endswith('joint4'):
        r.set_joint_angle(j.replace('4','2'),-1)
    while True:
        joint_curr=r.get_joint_angle(j,True)
        if joint_curr>joint_max:
            joint_max=joint_curr
            r.set_joint_angle(j,joint_curr+0.1)
            time.sleep(0.1)
        else:
            r.set_joint_angle(j,0)
            break
    print('Looking for min {}...'.format(j))
    while True:
        joint_curr=r.get_joint_angle(j,True)
        if joint_curr<joint_min:
            joint_min=joint_curr
            r.set_joint_angle(j,joint_curr-0.1)
            time.sleep(0.1)
        else:
            r.set_joint_angle(j,0)
            break
    joints_range[j]=(joint_min,joint_max)

joints_range

for j in joints_range:
    min_joint,max_joint=joints_range[j]
    print('{}: ({}, {})'.format(j, min_joint * 180 / math.pi,
                       max_joint * 180 / math.pi))

r.get_ultrasonic_sensor_reading(3)

#Stop the simulation
api.simulation.stop()

#Close the scene
vrep.simxCloseScene(api.simulation._id,vrep.simx_opmode_blocking)

#Close the connection to the simulator
api.close_connection()



