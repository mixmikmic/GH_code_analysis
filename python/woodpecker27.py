import GCode
import GRBL

cnc = GRBL.GRBL(port="/dev/cnc_3018")

cnc.status

import configparser
cfg = configparser.ConfigParser()
cfg.read("grbl_config.ini")

machine_settings = cfg["CNC_3018"]
for setting, value in machine_settings.items():
    r = cnc.cmd("{}={}".format(setting, value))
    print(r)

cnc.cmd("G21")

cnc.cmd("G91")

cnc.cmd("G0 F10 X2")

cnc.cmd("G0 F10 Y2")

cnc.cmd("G0 F10 Z-2")

cnc.laser_mode=1

cnc.cmd("M3 S1")



