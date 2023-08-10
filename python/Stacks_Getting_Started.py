get_ipython().system('pip3 install --user git+https://bitbucket.org/subinitial/subinitial.git')

import subinitial.stacks as stacks
print("Stacks Library Major Version:", stacks.VERSION_STACKS[0])

get_ipython().system('ping 192.168.1.49')

import subinitial.stacks1 as stacks
core = stacks.Core(host="192.168.1.49")
core.print_console("id")

