get_ipython().system('pip install h5py')

get_ipython().system("sed -i.bak '/run_tests/d' /usr/local/lib/python2.7/dist-packages/h5py/__init__.py")

import h5py

get_ipython().system('pip install keras')



