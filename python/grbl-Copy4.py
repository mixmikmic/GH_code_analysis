get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')

get_ipython().run_cell_magic('file', 'GRBL.py', 'import serial\nclass GRBL(object):\n    BAUDRATE = 115200\n    \n    def __init__(self, port):\n        self.serial = serial.Serial(port=port,\n                                    baudrate=GRBL.BAUDRATE,\n                                    timeout=0.10)\n        \n    def write(self, command_line=""):\n        self.serial.flushInput()\n        self.serial.write("\\n".encode())\n        self.serial.write("{cmd}\\n".format(cmd=command_line).encode())\n        \n    def read(self, multiline=True):\n        if multiline:\n            responses = self.serial.readlines()\n            responses = [response.decode().strip() for response in responses]\n            return responses\n        else:\n            response = self.serial.readline()\n            return response.decode().strip()\n \n    def cmd(self, command_line, resp=True, multiline=True):\n        self.write(command_line)\n        if resp:\n            return self.read(multiline=multiline)\n        return None\n    \n    def reset(self):\n        ret = self.cmd("\\x18")\n        assert(ret[-1]==\'ok\')\n        \n    def sleep(self):\n        ret = self.cmd("\\x18")\n        assert(ret[-1]==\'ok\')\n    \n    def kill_alarm(self):\n        ret = self.cmd("$X")\n        assert(ret[-1]==\'ok\')\n        \n    def home(self):\n        self.write("$H")\n        \n\nsettings = [\n    ("$0", "step_pulse"),\n    ("$1", "step_idle_delay"),\n    ("$2", "step_port_invert"),\n    ("$3", "direction_port_invert"),\n    ("$4", "step_enable_invert"),\n    ("$5", "limit_pin_invert"),\n    ("$6", "probe_pin_invert"),\n    ("$10", "status_report"),\n    ("$11", "junction_deviation"),\n    ("$12", "arc_tolerance"),\n    ("$13", "report_inches"),\n    ("$20", "soft_limits"),\n    ("$21", "hard_limits"),\n    ("$22", "homing_cycle"),\n    ("$23", "homing_dir_invert"),\n    ("$24", "homing_feed"),\n    ("$25", "homing_seek"),\n    ("$26", "homing_debounce"),\n    ("$27", "homing_pull_off"),\n    ("$30", "max_spindle_speed"),\n    ("$31", "min_spindle_speed"),\n    ("$32", "laser_mode"),\n    ("$100", "x_steps_mm"),\n    ("$101", "y_steps_mm"),\n    ("$102", "z_steps_mm"),\n    ("$110", "x_max_rate"),\n    ("$111", "y_max_rate"),\n    ("$112", "z_max_rate"),\n    ("$120", "x_acceleration"),\n    ("$121", "y_acceleration"),\n    ("$122", "z_acceleration"),\n    ("$130", "x_travel"),\n    ("$131", "y_travel"),\n    ("$132", "z_travel"),\n    ]\n\ndef grbl_getter_generator(cmd):\n    def grbl_getter(self):\n        config = self.cmd("$$", resp=True, multiline=True)\n        for config_line in config:\n            if config_line.startswith("$"):\n                key, value = config_line.split("=")\n                if key == cmd:\n                    return float(value)\n        return None\n    return grbl_getter\n    \ndef grbl_setter_generator(cmd):\n    def grbl_setter(self, value):\n        set_cmd = "{cmd}={value}".format(cmd=cmd, value=value)\n        ret = self.cmd(set_cmd, resp=True, multiline=False)\n        print(ret)\n        \n    return grbl_setter\n\nfor setting in settings:\n    cmd = setting[0]\n    name = setting[1]\n    \n    setter = grbl_setter_generator(cmd)\n    getter = grbl_getter_generator(cmd)\n    \n    prop = property(fget=getter,\n                    fset=setter,\n                    doc=" ".join(name.split("_")))\n    \n    setattr(GRBL, name, prop)')

get_ipython().run_line_magic('aimport', 'GRBL')

grbl = GRBL.GRBL("/dev/cnc_3018")

grbl.cmd("?")

grbl.cmd("$?")

gcode_parameters = grbl.cmd("$#") # View gcode parameters
gcode_parameters

def gcode_param_gen(parameter):
    def gcode_param(self):
        gcode_parameters = self.cmd("$#") # View gcode parameters
        for gcode_parameter in gcode_parameters: 
            if parameter in gcode_parameter:
                _, value = gcode_parameter.split(":")
                value = value.strip("]")
                values = value.split(",")
                values = [float(value) for value in values]
                
                return values
        return None
    return gcode_param

G54 = gcode_param_gen("G54")

G54(grbl)

print("gcode_parameters = [")
for p in gcode_parameters:
    if p.startswith("["):
        k, *_ = p.split(":")
        k = k.strip("[")
        print("    \"{}\",".format(k))
print("    ]")

gcode_parameters = [
    "G54",
    "G55",
    "G56",
    "G57",
    "G58",
    "G59",
    "G28",
    "G30",
    "G92",
    "TLO",
    "PRB",
    ]

for parameter in gcode_parameters:
    fcn = gcode_param_gen(parameter)
    prop = property(fget=fcn)
    setattr(GRBL.GRBL, parameter, prop)

grbl = GRBL.GRBL("/dev/cnc_3018")

grbl.G54

def status(self):
    ret = self.cmd("")

