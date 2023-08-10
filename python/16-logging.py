# https://python.swaroopch.com/stdlib.html

import os
import platform
import logging

if platform.platform().startswith('Windows'):
    logging_file = os.path.join(os.getenv('HOMEDRIVE'),
                                os.getenv('HOMEPATH'),
                                'test.log')
    print('This is a Windows machine')
else:
    logging_file = os.path.join(os.getenv('HOME'),
                                'test.log')
    print('This is a Unix machine')

print("Logging to", logging_file)

logging.basicConfig(
    level=logging.INFO, # DEBUG > INFO > WARNING
    format='%(asctime)s : %(levelname)s : %(message)s',
    filename=logging_file,
    filemode='w',
)

logging.debug("This is a debugging message")
logging.warning("This is a warning message")
logging.info("This is a information message")

get_ipython().system(' cat $HOME/test.log')

get_ipython().system(' cat $HOME/test.log | grep INFO')

