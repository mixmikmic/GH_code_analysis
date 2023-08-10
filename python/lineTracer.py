#Constants for accessing visionSensor readings
visionSensor_intensity_min=0
visionSensor_red_min=1
visionSensor_green_min=2
visionSensor_blue_min=3
visionSensor_depth_min=4

visionSensor_intensity_max=5
visionSensor_red_max=6
visionSensor_green_max=7
visionSensor_blue_max=8
visionSensor_depth_max=9

visionSensor_intensity_av=10
visionSensor_red_av=11
visionSensor_green_av=12
visionSensor_blue_av=13
visionSensor_depth_av=14

rclass='lineTracer'
print('Loading class: {}'.format(rclass))

from pyrep.vrep.vrep import simxGetFloatSignal
from pyrep.vrep import vrep as v

class lineTracer:

    def __init__(self, api: VRep):
        self._api = api
        self._left_motor = api.joint.with_velocity_control("DynamicLeftJoint")
        self._right_motor = api.joint.with_velocity_control("DynamicRightJoint")
        self._left_sensor = api.sensor.vision("LeftSensor")
        self._right_sensor = api.sensor.vision("RightSensor")
        self.id = api._id
    
    #The following function will return the total accumulated angle turned by the left wheel
    def getval(self):
        return simxGetFloatSignal(self.id,'leftEncoder',v.simx_opmode_streaming)#simx_opmode_streaming, simx_opmode_buffer)

    #The following function will return the total accumulated rotation count of the left wheel
    def getrots(self):
        return simxGetFloatSignal(self.id,'leftEncoder_rots',v.simx_opmode_streaming)[1]#simx_opmode_buffer)

    def set_two_motor(self, left: float, right: float):
        self._left_motor.set_target_velocity(left)
        self._right_motor.set_target_velocity(right)

    def stop(self):
        self.set_two_motor(0, 0)

    def fwd_right(self, speed=20.0):
        self.set_two_motor(speed, speed/2)

    def fwd_left(self, speed=20.0):
        self.set_two_motor(speed/2, speed)

    def move_forward(self, speed=20.0):
        self.set_two_motor(speed, speed)

    def move_backward(self, speed=20.0):
        self.set_two_motor(-speed, -speed)

    def rotate_right(self, speed=2.0):
        self.set_two_motor(speed, -speed)

    def rotate_left(self, speed=2.0):
        self.set_two_motor(-speed, speed)

    
    def right_line(self):
        return self._left_sensor.read()[0]
    
    def left_line(self):
        return self._left_sensor.read()[0]
    
    def right_light(self):
        return self._right_sensor.read()[2][0][visionSensor_intensity_av]
    
    def left_light(self):
        return self._left_sensor.read()[2][0][visionSensor_intensity_av]

    def position_left_joint(self):
        return self._left_motor.get_position()
    
methods = [method for method in dir(eval(rclass)) if not method.startswith('_')]
print('Methods available in {}:\n\t{}\n'.format(eval(rclass).__name__ , '\n\t'.join(methods)))

