import time
import pep8

time.strftime("last updated %a, %d %b %Y %H:%M", time.localtime())

get_ipython().system('pep8 --first tools.py')
#!pep8 --show-source --show-pep8 tools.py

