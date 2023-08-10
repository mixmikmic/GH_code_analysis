# Load in the base Overlay
from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import PMOD_GROVE_G3
from pynq.iop import PMOD_GROVE_G4
from pynq.iop import Pmod_IIC

basic_font = [[0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
[0x00,0x00,0x5F,0x00,0x00,0x00,0x00,0x00],
[0x00,0x00,0x07,0x00,0x07,0x00,0x00,0x00],
[0x00,0x14,0x7F,0x14,0x7F,0x14,0x00,0x00],
[0x00,0x24,0x2A,0x7F,0x2A,0x12,0x00,0x00],
[0x00,0x23,0x13,0x08,0x64,0x62,0x00,0x00],
[0x00,0x36,0x49,0x55,0x22,0x50,0x00,0x00],
[0x00,0x00,0x05,0x03,0x00,0x00,0x00,0x00],
[0x00,0x1C,0x22,0x41,0x00,0x00,0x00,0x00],
[0x00,0x41,0x22,0x1C,0x00,0x00,0x00,0x00],
[0x00,0x08,0x2A,0x1C,0x2A,0x08,0x00,0x00],
[0x00,0x08,0x08,0x3E,0x08,0x08,0x00,0x00],
[0x00,0xA0,0x60,0x00,0x00,0x00,0x00,0x00],
[0x00,0x08,0x08,0x08,0x08,0x08,0x00,0x00],
[0x00,0x60,0x60,0x00,0x00,0x00,0x00,0x00],
[0x00,0x20,0x10,0x08,0x04,0x02,0x00,0x00],
[0x00,0x3E,0x51,0x49,0x45,0x3E,0x00,0x00],
[0x00,0x00,0x42,0x7F,0x40,0x00,0x00,0x00],
[0x00,0x62,0x51,0x49,0x49,0x46,0x00,0x00],
[0x00,0x22,0x41,0x49,0x49,0x36,0x00,0x00],
[0x00,0x18,0x14,0x12,0x7F,0x10,0x00,0x00],
[0x00,0x27,0x45,0x45,0x45,0x39,0x00,0x00],
[0x00,0x3C,0x4A,0x49,0x49,0x30,0x00,0x00],
[0x00,0x01,0x71,0x09,0x05,0x03,0x00,0x00],
[0x00,0x36,0x49,0x49,0x49,0x36,0x00,0x00],
[0x00,0x06,0x49,0x49,0x29,0x1E,0x00,0x00],
[0x00,0x00,0x36,0x36,0x00,0x00,0x00,0x00],
[0x00,0x00,0xAC,0x6C,0x00,0x00,0x00,0x00],
[0x00,0x08,0x14,0x22,0x41,0x00,0x00,0x00],
[0x00,0x14,0x14,0x14,0x14,0x14,0x00,0x00],
[0x00,0x41,0x22,0x14,0x08,0x00,0x00,0x00],
[0x00,0x02,0x01,0x51,0x09,0x06,0x00,0x00],
[0x00,0x32,0x49,0x79,0x41,0x3E,0x00,0x00],
[0x00,0x7E,0x09,0x09,0x09,0x7E,0x00,0x00],
[0x00,0x7F,0x49,0x49,0x49,0x36,0x00,0x00],
[0x00,0x3E,0x41,0x41,0x41,0x22,0x00,0x00],
[0x00,0x7F,0x41,0x41,0x22,0x1C,0x00,0x00],
[0x00,0x7F,0x49,0x49,0x49,0x41,0x00,0x00],
[0x00,0x7F,0x09,0x09,0x09,0x01,0x00,0x00],
[0x00,0x3E,0x41,0x41,0x51,0x72,0x00,0x00],
[0x00,0x7F,0x08,0x08,0x08,0x7F,0x00,0x00],
[0x00,0x41,0x7F,0x41,0x00,0x00,0x00,0x00],
[0x00,0x20,0x40,0x41,0x3F,0x01,0x00,0x00],
[0x00,0x7F,0x08,0x14,0x22,0x41,0x00,0x00],
[0x00,0x7F,0x40,0x40,0x40,0x40,0x00,0x00],
[0x00,0x7F,0x02,0x0C,0x02,0x7F,0x00,0x00],
[0x00,0x7F,0x04,0x08,0x10,0x7F,0x00,0x00],
[0x00,0x3E,0x41,0x41,0x41,0x3E,0x00,0x00],
[0x00,0x7F,0x09,0x09,0x09,0x06,0x00,0x00],
[0x00,0x3E,0x41,0x51,0x21,0x5E,0x00,0x00],
[0x00,0x7F,0x09,0x19,0x29,0x46,0x00,0x00],
[0x00,0x26,0x49,0x49,0x49,0x32,0x00,0x00],
[0x00,0x01,0x01,0x7F,0x01,0x01,0x00,0x00],
[0x00,0x3F,0x40,0x40,0x40,0x3F,0x00,0x00],
[0x00,0x1F,0x20,0x40,0x20,0x1F,0x00,0x00],
[0x00,0x3F,0x40,0x38,0x40,0x3F,0x00,0x00],
[0x00,0x63,0x14,0x08,0x14,0x63,0x00,0x00],
[0x00,0x03,0x04,0x78,0x04,0x03,0x00,0x00],
[0x00,0x61,0x51,0x49,0x45,0x43,0x00,0x00],
[0x00,0x7F,0x41,0x41,0x00,0x00,0x00,0x00],
[0x00,0x02,0x04,0x08,0x10,0x20,0x00,0x00],
[0x00,0x41,0x41,0x7F,0x00,0x00,0x00,0x00],
[0x00,0x04,0x02,0x01,0x02,0x04,0x00,0x00],
[0x00,0x80,0x80,0x80,0x80,0x80,0x00,0x00],
[0x00,0x01,0x02,0x04,0x00,0x00,0x00,0x00],
[0x00,0x20,0x54,0x54,0x54,0x78,0x00,0x00],
[0x00,0x7F,0x48,0x44,0x44,0x38,0x00,0x00],
[0x00,0x38,0x44,0x44,0x28,0x00,0x00,0x00],
[0x00,0x38,0x44,0x44,0x48,0x7F,0x00,0x00],
[0x00,0x38,0x54,0x54,0x54,0x18,0x00,0x00],
[0x00,0x08,0x7E,0x09,0x02,0x00,0x00,0x00],
[0x00,0x18,0xA4,0xA4,0xA4,0x7C,0x00,0x00],
[0x00,0x7F,0x08,0x04,0x04,0x78,0x00,0x00],
[0x00,0x00,0x7D,0x00,0x00,0x00,0x00,0x00],
[0x00,0x80,0x84,0x7D,0x00,0x00,0x00,0x00],
[0x00,0x7F,0x10,0x28,0x44,0x00,0x00,0x00],
[0x00,0x41,0x7F,0x40,0x00,0x00,0x00,0x00],
[0x00,0x7C,0x04,0x18,0x04,0x78,0x00,0x00],
[0x00,0x7C,0x08,0x04,0x7C,0x00,0x00,0x00],
[0x00,0x38,0x44,0x44,0x38,0x00,0x00,0x00],
[0x00,0xFC,0x24,0x24,0x18,0x00,0x00,0x00],
[0x00,0x18,0x24,0x24,0xFC,0x00,0x00,0x00],
[0x00,0x00,0x7C,0x08,0x04,0x00,0x00,0x00],
[0x00,0x48,0x54,0x54,0x24,0x00,0x00,0x00],
[0x00,0x04,0x7F,0x44,0x00,0x00,0x00,0x00],
[0x00,0x3C,0x40,0x40,0x7C,0x00,0x00,0x00],
[0x00,0x1C,0x20,0x40,0x20,0x1C,0x00,0x00],
[0x00,0x3C,0x40,0x30,0x40,0x3C,0x00,0x00],
[0x00,0x44,0x28,0x10,0x28,0x44,0x00,0x00],
[0x00,0x1C,0xA0,0xA0,0x7C,0x00,0x00,0x00],
[0x00,0x44,0x64,0x54,0x4C,0x44,0x00,0x00],
[0x00,0x08,0x36,0x41,0x00,0x00,0x00,0x00],
[0x00,0x00,0x7F,0x00,0x00,0x00,0x00,0x00],
[0x00,0x41,0x36,0x08,0x00,0x00,0x00,0x00],
[0x00,0x02,0x01,0x01,0x02,0x01,0x00,0x00],
[0x00,0x02,0x05,0x05,0x02,0x00,0x00,0x00]]

class Python_Grove_OLED(Pmod_IIC):
    """This class controls the Grove OLED.
    
    This class inherits from the PMODIIC class.
    
    Attributes
    ----------
    iop : _IOP
        The _IOP object returned from the DevMode.
    scl_pin : int
        The SCL pin number.
    sda_pin : int
        The SDA pin number.
    iic_addr : int
        The IIC device address.
    
    """
    def __init__(self, pmod_id, gr_pins): 
        """Return a new instance of a grove OLED object. 
    
        Note
        ----
        
        Parameters
        ----------
        pmod_id : int
            The PMOD ID (1, 2) corresponding to (PMODA, PMODB).
        gr_pins: list
            Adapter pins selected.
            
        """
        if gr_pins in [PMOD_GROVE_G3,PMOD_GROVE_G4]:
            [scl_pin,sda_pin] = gr_pins
        else:
            raise ValueError("Valid Grove Pins are on G3 or G4.")
        
        super().__init__(pmod_id, scl_pin, sda_pin, 0x3C)
        
        # Unlock OLED driver IC MCU interface
        self._send_cmd(0xFD) 
        self._send_cmd(0x12)
        # Set display off
        self._send_cmd(0xAE)
        # Switch on display
        self._send_cmd(0xAF) 
        self._send_cmd(0xA4)
        
    def _send_cmd(self, word):
        """Send a command to the IIC driver.
        
        This method relies on the send() in the parent class.
        
        Parameters
        ----------
        word : int
            A 32-bit command word to be written to the driver.
            
        Returns
        -------
        None
        
        """
        self.send([0x80,word])
        
    def _send_data(self, word):
        """Send a command to the IIC driver.
        
        This method relies on the send() in the parent class.
        
        Parameters
        ----------
        word : int
            A 32-bit data word to be written to the driver.
            
        Returns
        -------
        None
        
        """
        self.send([0x40,word])
    
    def set_normal_mode(self):
        """Set the display mode to 'normal'.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        self._send_cmd(0xA4)
    
    def set_inverse_mode(self):
        """Set the display mode to 'inverse'.
        
        This mode has white background and black characters.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        self._send_cmd(0xA7)
    
    def _put_char(self, chr):
        """Print a single character on the OLED screen.
        
        Note
        ----
        This method is only for internal use of this class. To print strings
        or characters, users should use the write() method.
        
        Parameters
        ----------
        chr : str
            A string of length 1 to be put onto the screen.
            
        Returns
        -------
        None
        
        """
        global basic_font
        c_add=ord(chr)
        if c_add<32 or c_add>127:     
            # Ignore non-printable ASCII characters
            chr = ' '
            c_add=ord(chr)
        for j in range(8):
            self._send_data(basic_font[c_add-32][j])

    def set_XY(self, row, column):
        """Set the location where to start printing.
        
        Parameters
        ----------
        row : int
            The row number indicating where to start.
        column : int
            The column number indicating where to start.
            
        Returns
        -------
        None
        
        """
        self._send_cmd(0xB0 + row)
        self._send_cmd(0x00 + (8*column & 0x0F))
        self._send_cmd(0x10 + ((8*column>>4)&0x0F))
        
    def write(self, text):
        """Write the strings to the OLED screen.
        
        This is the method to be used when writing strings.
        
        Parameters
        ----------
        text : str
            A string to be put onto the screen.
            
        Returns
        -------
        None
        
        """
        for i in range(len(text)):
            self._put_char(text[i])
    
    def clear(self):
        """Clear the OLED screen.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        for i in range(8):
            self.set_XY(i,0)
            for j in range(16):  
                self._put_char(' ')
        self.set_XY(0,0)

from pynq import PL
from pynq.iop import PMODB
from pynq.iop import PMOD_GROVE_G3

# Flush IOP
PL.reset()

oled = Python_Grove_OLED(PMODB,PMOD_GROVE_G3)
oled.clear()
oled.write('Hi from Python.')
del oled

from pynq.iop import Grove_OLED
from pynq.iop import PMODB
from pynq.iop import PMOD_GROVE_G3

# Flush IOP
PL.reset()

oled = Grove_OLED(PMODB,PMOD_GROVE_G3)
oled.clear()
oled.write('Hello from      Microblaze.')
del oled



