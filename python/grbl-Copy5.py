get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')

get_ipython().run_cell_magic('file', 'GRBL.py', 'import serial\nclass GRBL(object):\n    BAUDRATE = 115200\n    \n    def __init__(self, port):\n        self.serial = serial.Serial(port=port,\n                                    baudrate=GRBL.BAUDRATE,\n                                    timeout=0.10)\n        \n    def write(self, command_line=""):\n        self.serial.flushInput()\n        self.serial.write("\\n".encode())\n        self.serial.write("{cmd}\\n".format(cmd=command_line).encode())\n        \n    def read(self, multiline=True):\n        if multiline:\n            responses = self.serial.readlines()\n            responses = [response.decode().strip() for response in responses]\n            return responses\n        else:\n            response = self.serial.readline()\n            return response.decode().strip()\n \n    def cmd(self, command_line, resp=True, multiline=True):\n        self.write(command_line)\n        if resp:\n            return self.read(multiline=multiline)\n        return None\n    \n    def reset(self):\n        """ https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands#grbl-v11-realtime-commands\n        """\n        ret = self.cmd("\\x18")\n        assert(ret[-1]==\'ok\')\n        \n    def sleep(self):\n        """ https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands#slp---enable-sleep-mode\n        """\n        ret = self.cmd("$SLP")\n        assert(ret[-1]==\'ok\')\n    \n    @property\n    def status(self):\n        """\n        """\n        ret = self.cmd("?")\n        assert(ret[-1]==\'ok\')\n        return ret[1]\n        \n    def kill_alarm(self):\n        """ https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands#x---kill-alarm-lock\n        """\n        ret = self.cmd("$X")\n        assert(ret[-1]==\'ok\')\n        \n    def home(self):\n        """ https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands#h---run-homing-cycle\n        """\n        self.write("$H")\n        assert(ret[-1]==\'ok\')\n        \n# https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration#---view-grbl-settings\nsettings = [\n    ("$0", "step_pulse"),\n    ("$1", "step_idle_delay"),\n    ("$2", "step_port_invert"),\n    ("$3", "direction_port_invert"),\n    ("$4", "step_enable_invert"),\n    ("$5", "limit_pin_invert"),\n    ("$6", "probe_pin_invert"),\n    ("$10", "status_report"),\n    ("$11", "junction_deviation"),\n    ("$12", "arc_tolerance"),\n    ("$13", "report_inches"),\n    ("$20", "soft_limits"),\n    ("$21", "hard_limits"),\n    ("$22", "homing_cycle"),\n    ("$23", "homing_dir_invert"),\n    ("$24", "homing_feed"),\n    ("$25", "homing_seek"),\n    ("$26", "homing_debounce"),\n    ("$27", "homing_pull_off"),\n    ("$30", "max_spindle_speed"),\n    ("$31", "min_spindle_speed"),\n    ("$32", "laser_mode"),\n    ("$100", "x_steps_mm"),\n    ("$101", "y_steps_mm"),\n    ("$102", "z_steps_mm"),\n    ("$110", "x_max_rate"),\n    ("$111", "y_max_rate"),\n    ("$112", "z_max_rate"),\n    ("$120", "x_acceleration"),\n    ("$121", "y_acceleration"),\n    ("$122", "z_acceleration"),\n    ("$130", "x_travel"),\n    ("$131", "y_travel"),\n    ("$132", "z_travel"),\n    ]\n\ndef grbl_getter_generator(cmd):\n    def grbl_getter(self):\n        config = self.cmd("$$", resp=True, multiline=True)\n        for config_line in config:\n            if config_line.startswith("$"):\n                key, value = config_line.split("=")\n                if key == cmd:\n                    return float(value)\n        return None\n    return grbl_getter\n    \ndef grbl_setter_generator(cmd):\n    def grbl_setter(self, value):\n        set_cmd = "{cmd}={value}".format(cmd=cmd, value=value)\n        ret = self.cmd(set_cmd, resp=True, multiline=False)\n        print(ret)\n        \n    return grbl_setter\n\nfor setting in settings:\n    cmd = setting[0]\n    name = setting[1]\n    \n    setter = grbl_setter_generator(cmd)\n    getter = grbl_getter_generator(cmd)\n    \n    prop = property(fget=getter,\n                    fset=setter,\n                    doc=" ".join(name.split("_")))\n    \n    setattr(GRBL, name, prop)\n    \n# https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands#---view-gcode-parameters\ngcode_parameters = [\n    "G54",\n    "G55",\n    "G56",\n    "G57",\n    "G58",\n    "G59",\n    "G28",\n    "G30",\n    "G92",\n    "TLO",\n    "PRB",\n    ]\n\n\ndef gcode_param_gen(parameter):\n    def gcode_param(self):\n        gcode_parameters = self.cmd("$#") # View gcode parameters\n        for gcode_parameter in gcode_parameters: \n            if parameter in gcode_parameter:\n                _, value = gcode_parameter.split(":")\n                value = value.strip("]")\n                values = value.split(",")\n                values = [float(value) for value in values]\n                \n                return values\n        return None\n    return gcode_param\n\n\nfor parameter in gcode_parameters:\n    fcn = gcode_param_gen(parameter)\n    prop = property(fget=fcn)\n    setattr(GRBL, parameter, prop)')

get_ipython().run_line_magic('aimport', 'GRBL')

grbl = GRBL.GRBL("/dev/cnc_3018")

grbl.reset()

# Metric
grbl.cmd("G21")
# Relative
grbl.cmd("G91")

grbl.cmd("M3 S1000")

grbl.cmd("G0 F10")

grbl.cmd("G0 Z-2")

grbl.cmd("G0 Z2")

grbl.cmd("G0 Z2")

grbl.cmd("G1 F500")

grbl.cmd("G1 X10")
grbl.cmd("G1 Y10")
grbl.cmd("G1 X-10")
grbl.cmd("G1 Y-10")

grbl.cmd("G1 X100")
grbl.cmd("G1 Y100")
grbl.cmd("G1 X-100")
grbl.cmd("G1 Y-100")

grbl.cmd("G1 X-200")
grbl.cmd("G1 Y-200")
grbl.cmd("G1 X200")
grbl.cmd("G1 Y200")

grbl.cmd("G1 X100 Y100")

grbl.cmd("G1 Y-100")

grbl.cmd("G1 Z0.5")

grbl.cmd("G1 Y100")

grbl.cmd("G1 Z-0.5")

grbl.cmd("G3 X0 Y-100 J-50 I0")

grbl.cmd("G0 Z-5")

grbl.cmd("G0 X200")

grbl.cmd("G1 Z5")

grbl.cmd("G1 Z2")

grbl.cmd("G3 X0 Y-50 J-25 I0")
grbl.cmd("G3 X0 Y50 J25 I0")

grbl.cmd("M5")

grbl.cmd("G0 F750")

grbl.cmd("G0 Z-10")

grbl.x_steps_mm /(200/62.71)

grbl.x_steps_mm = 1



