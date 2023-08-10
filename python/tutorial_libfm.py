get_ipython().magic('cat data/libfm/train.libfm')

get_ipython().magic('cat data/libfm/test.libfm')

bin_dir = '/Users/zhangjun/Documents/libfm-1.42.src/bin'
get_ipython().magic('ls $bin_dir')

import os
script_dir = os.path.join(bin_dir, "libFM")
get_ipython().magic('ls $script_dir')

import sys
from subprocess import call
try:
    command_str =         "%s -task r -method mcmc -train data/libfm/train.libfm -test data/libfm/test.libfm -iter 10 -dim '1,1,2' -out data/libfm/output.libfm"         % script_dir
    command_list = command_str.split(' ')
    retcode = call(command_list)
    if retcode < 0:
        print >>sys.stderr, "Child was terminated by signal", -retcode
    else:
        print >>sys.stderr, "Child returned", retcode
except OSError as e:
    print >>sys.stderr, "Execution failed:", e

get_ipython().magic('cat data/libfm/output.libfm')

