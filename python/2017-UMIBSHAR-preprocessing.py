## Readme File ##
from IPython.display import IFrame
IFrame("./2017-UMIBSHAR-Readme.txt", width=900, height=900)

### Read File
import scipy.io as sio
mat_contents = sio.loadmat('C:\\Users\\paulo\\Documents\\py-har\\datasets\\UMIBSHAR-2017\\data\\full_data.mat')
mat_contents

