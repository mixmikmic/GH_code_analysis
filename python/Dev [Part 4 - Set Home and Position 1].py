get_ipython().run_line_magic('pylab', 'inline')
import sys
from ipywidgets import interact

sys.path.append('src')
from motorControl import *
from trajectoryPlanning import *

rc = connect(portName = "/dev/tty.usbserial-A9ETDN3N")

#Initialize motor objects for each motor:
vL = Motor(address = 0x81, motorNumber = 2, rc = rc, signFlipped = False,            motorCounter = 0, kPID = [1e-2, 1.0])
vR = Motor(address = 0x81, motorNumber = 1, rc = rc, signFlipped = False,            motorCounter = 1, kPID = [1e-2, 1.0])
LR = Motor(address = 0x80, motorNumber = 1, rc = rc, signFlipped = False, 
           motorCounter = 2, kPID = [1e-2, 1.0])
FB = Motor(address = 0x80, motorNumber = 2, rc = rc, signFlipped = True,            motorCounter = 3, kPID = [1e-2, 1.0])
yaw = Motor(address = 0x82, motorNumber = 2, rc = rc, signFlipped = False,             motorCounter = 4, kPID = [1e-2, 1.0])
pitch = Motor(address = 0x82, motorNumber = 1, rc = rc, signFlipped = False,               motorCounter = 5, kPID = [1e-2, 1.0])

#Keep in a nice motor list:
motors = [vL, vR, LR, FB, yaw, pitch]

savePositions(motors)
getPositions(motors)

interact(manualControl, leftUD = (-50, 50), rightUD = (-50, 50), leftRight = (-50, 50),          fB = (-50, 50), tilt = (-50, 50), pan = (-50, 50))

stopAll(rc)
savePositions(motors)
getPositions(motors)

# for motor in motors:
#     motor.resetEncoders()

# f = open('savedPositions/home.p', 'wb')
# pickle.dump(getPositions(motors), f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

# f = open('savedPositions/position2.p', 'wb')
# pickle.dump(getPositions(motors), f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

# f = open('savedPositions/position1.p', 'wb')
# pickle.dump(getPositions(motors), f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

f = open('savedPositions/home.p', 'r')
home = pickle.load(f)
f.close()

f = open('savedPositions/position1.p', 'r')
position1 = pickle.load(f)
f.close()

print position1, home



targetPositions = position1
totalTime = 40.0
rampTime = 5.0

startingPositions = getPositions(motors)
lookAheadTime = 1.0
tolerance = 100.0 #Anything less than this many ticks we're calling "Not a move"

motorsToMove = []
targetPositionsToMove = []
for i, motor in enumerate(motors):
    if abs(motor.getPosition()-targetPositions[i]) > tolerance:
        motorsToMove.append(motor)
        targetPositionsToMove.append(targetPositions[i])
        
trajectories = []
for i, motor in enumerate(motorsToMove):
    trajectories.append(SimpleQuadraticTrajectory(tu = rampTime, tt = totalTime,                                             p1 = motor.getPosition(), p2 = targetPositionsToMove[i]))

for motor in motorsToMove:
    motor.initialize(targetVelocityMin = -2500.0, targetVelocityMax = 2500.0)
    motor.clearTracking()
    
startTime = time.time()
timeElapsed = 0.0
    
while timeElapsed < totalTime:
    timeElapsed = time.time()-startTime
    
    for i, motor in enumerate(motorsToMove):
        lookAheadValue = trajectories[i].compute(timeElapsed + lookAheadTime)
        motor.controlledMove(targetPosition = lookAheadValue, timeToReach = lookAheadTime)

stopAll(rc)
savePositions(motors)
print getPositions(motors), targetPositions
print getPositions(motors) - np.array(targetPositions)





