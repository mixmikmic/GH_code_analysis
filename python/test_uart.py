from pynq.overlays.base import BaseOverlay

base = BaseOverlay('base.bit')

get_ipython().run_cell_magic('microblaze', 'base.ARDUINO', '#include <uart.h>\n#include <string.h>\n#include <pyprintf.h>\n\nint test_arduino() {\n    uart device = uart_open(1, 0);\n    if (device<0){\n        return -2;\n    }\n    int len = 14;\n    unsigned char message1[len];\n    strncpy(message1, "hello world123", len);\n    unsigned char message2[len];\n    uart_write(device, message1, len);\n    delay_ms(15);\n    uart_read(device, message2, len);\n    pyprintf("Received\\n");\n    int i;\n    for (i=0;i<len;i++){\n        pyprintf("%c", message2[i]);\n        if (message2[i]!=message1[i]){\n            return -1;\n        }\n    }\n    return 0;\n}')

test_arduino()

get_ipython().run_cell_magic('microblaze', 'base.ARDUINO', '#include <xio_switch.h>\n\nint get_pin(){\n    set_pin(1, UART0_TX);\n    set_pin(0, UART0_RX);\n    return Xil_In32(SWITCH_BASEADDR+4*0);\n}')

get_pin()

get_ipython().run_cell_magic('microblaze', 'base.RPI', '#include <uart.h>\n#include <string.h>\n#include <pyprintf.h>\n\nint test_rpi() {\n    uart device = uart_open(14, 15);\n    if (device<0){\n        return -2;\n    }\n    int len = 12;\n    unsigned char message1[len];\n    strncpy(message1, "hello world!", len);\n    unsigned char message2[len];\n    uart_write(device, message1, len);\n    delay_ms(13);\n    uart_read(device, message2, len);\n    pyprintf("Received\\n");\n    int i;\n    for (i=0;i<len;i++){\n        pyprintf("%c", message2[i]);\n        if (message2[i]!=message1[i]){\n            return -1;\n        }\n    }\n    return 0;\n}')

test_rpi()

