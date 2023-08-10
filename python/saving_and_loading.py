import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b['incl@orbit'] = 56.789

print b.save('test.phoebe')

get_ipython().system('head -n 30 test.phoebe')

b2 = phoebe.Bundle.open('test.phoebe')

print b2.get_value('incl@orbit')

b = phoebe.Bundle.from_legacy('legacy.phoebe')

b.export_legacy('legacy_export.phoebe')

get_ipython().system('head -n 30 legacy_export.phoebe')

