from pynq import Overlay
from pynq.lib import Pmod_OLED

logictools = Overlay("logictools.bit")

logictools.pmoda.load(Pmod_OLED)
pmod_oled = logictools.pmoda

pmod_oled.clear()
pmod_oled.write('Welcome to the\nPynq-Z1 board!')

pmod_oled.clear()
pmod_oled.write('Python and Zynq\nproductivity &  performance')

def get_ip_address():
    ipaddr_slist = get_ipython().getoutput('hostname -I')
    ipaddr = (ipaddr_slist.s).split(" ")[0]
    return str(ipaddr)

pmod_oled.clear()
pmod_oled.write(get_ip_address())

