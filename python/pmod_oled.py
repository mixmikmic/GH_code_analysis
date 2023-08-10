from pynq import Overlay
from pynq.iop import Pmod_OLED
from pynq.iop import PMODA

ol = Overlay("base.bit")
ol.download()

pmod_oled = Pmod_OLED(PMODA)

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

