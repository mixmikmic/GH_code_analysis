import GCode
import GRBL

cnc = GRBL.GRBL(port="/dev/cnc_3018")

print("Laser Mode: {}".format(cnc.laser_mode))

from enum import IntEnum
class LaserPower(IntEnum):
    CONSTANT = 0
    DYNAMIC = 1
    
def init(power = LaserPower(0), feed = 200, laser = 25):
    program = GCode.GCode()
    program.G21() # Metric Units
    program.G91() # Rel positioning.
    program.G1(F=feed) # Set the feed rate
    program.G0() # But keep the laser off.
    if power==LaserPower.CONSTANT:
        program.M3(S=laser) # Laser settings
    else:
        program.M4(S=laser) # Laser settings
    return program

def end():
    program = GCode.GCode()
    program.M5() # Te
    return program

def heart(scale = 1):
    p = GCode.GCode()
    p.G0(X=2, Y=0)
    p.G1(X=-2, Y=2)
    p.G2(X=2, Y=0, I=1, J=0)
    p.G2(X=2, Y=0, I=1, J=0)
    p.G1(X=-2, Y=-2)
    return p

heart10 = heart(scale=1)
print(heart10)

cnc.run(init(laser=5)+heart(scale=1)+end())

cnc.run(init(laser=100)+heart(scale=1)+end())

def heart(scale = 1):
    p = GCode.GCode()
    p.G0(X=2*scale, Y=0)
    p.G1(X=-2*scale, Y=2*scale)
    p.G2(X=2*scale, Y=0, I=1*scale, J=0)
    p.G2(X=2*scale, Y=0, I=1*scale, J=0)
    p.G1(X=-2*scale, Y=-2*scale)
    return p

cnc.run(init(laser=100)+heart(scale=2)+end())

class SoftKill(Exception):
    pass

for scale in [4, 8, 16, 32, 65]:
    try:
        cnc.run(init(laser=100)+heart(scale=scale)+end())
        cnc.cmd("G1 X{}".format(scale)) # Move over to edge of heart
        cnc.cmd("G1 X10") # Move another 10
    except KeyboardInterrupt:
        cnc.cmd("!")
        raise(SoftKill("Keyboard"))

