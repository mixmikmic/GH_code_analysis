import sys; print('python:\t\t\t', str(sys.version_info[0]) + '.' + str(sys.version_info[1]))
print('modules:')
import numpy;   print('\tnumpy:\t\t', numpy.__version__)
import scipy;   print('\tscipy:\t\t', scipy.__version__)
import pandas;  print('\tpandas:\t\t', pandas.__version__)
import sklearn; print('\tsklearn:\t', sklearn.__version__)



