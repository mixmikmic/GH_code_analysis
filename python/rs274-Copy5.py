get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')

get_ipython().run_line_magic('aimport', 'GCode')
get_ipython().run_line_magic('aimport', 'GRBL')

cnc = GRBL.GRBL(port="/dev/cnc_3018")

cnc.laser_mode

def init(M3 = True, feed = 200, laser = 25):
    program = GCode.GCode()
    program.G21() # Metric Units
    program.G91() # Absolute positioning.
    program.G1(F=feed) #
    program.M3(S=laser) # Laser settings.
    return program

def end():
    program = GCode.GCode()
    program.M5() # Laser settings.
    return program

def square(size=20):    
    program = GCode.GCode()
    program.G1(X=size)
    program.G1(Y=size)
    program.G1(X=-size)
    program.G1(Y=-size)
    return program

program = init(M3=True, laser=1) + square() + end()
program

cnc.run(program);

cnc.cmd("M3 S1")

cnc.cmd("G0 X0") # Laser off

cnc.cmd("G1 X0") # Laser On

cnc.cmd("G0 Y+80")

def jogx(x=10):
    program = GCode.GCode()
    program.G0(X=x)
    return program

cnc.run(jogx(-20))

for laser in [1, 10, 50, 100, 150, 255, 1024]:
    print("\t"*3+"Lasers Set To: {}".format(laser))
    program = init(M3=True, laser=laser) + square(size=10) + end()
    cnc.run(program)
    cnc.run(jogx(20))



