# So we can import connectortools
import sys
sys.path.append('../')

import connectortools as ct

# Update to the latest neurods
get_ipython().system('pip install git+https://github.com/choldgraf/connectortools.git --user --upgrade')

# With a raw URL
ct.install_package(url='https://github.com/choldgraf/connectortools.git')

# With a package name
ct.install_package('sklearn')



