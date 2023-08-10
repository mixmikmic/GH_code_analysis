import sys
import logging
import argparse
import collections

def parse_args(cmd_line):
    Args = collections.namedtuple('Args',
                                  ['n', 'in_file'])
    n_trigger = False
    # default value for "n"
    n = 1
    for arg in cmd_line:
        if n_trigger:
            n = int(arg)
            n_trigger = False
            continue
        if arg == '-n':
            # next argument belongs to "-n"
            n_trigger = True
            continue
        else:
            # it must be the positional argument
            in_file = arg
    return Args(n=n, in_file=in_file)

# immaginary command line
cmd_line = '-n 5 myfile.txt'
parse_args(cmd_line.split())

# immaginary command line with multiple input files
cmd_line = '-n 5 myfile.txt another_one.txt'
parse_args(cmd_line.split())

def parse_args(cmd_line):
    Args = collections.namedtuple('Args',
                                  ['n',
                                   'verbose',
                                   'in_file',
                                   'another_file'])
    n_trigger = False
    # default value for "n"
    n = 1
    # default value for "verbose"
    verbose = False
    # list to hold the positional arguments
    positional = []
    for arg in cmd_line:
        if n_trigger:
            n = int(arg)
            n_trigger = False
            continue
        if arg == '-n':
            # next argument belongs to "-n"
            n_trigger = True
        elif arg == '--verbose' or arg == '-v':
            verbose = True
        else:
            # it must be the positional argument
            positional.append(arg)
    return Args(n=n,
                verbose=verbose,
                in_file=positional[0],
                another_file=positional[1])

# immaginary command line with multiple input files
cmd_line = '-n 5 myfile.txt another_one.txt'
parse_args(cmd_line.split())

def parse_args(cmd_line):
    Args = collections.namedtuple('Args',
                                  ['n',
                                   'verbose',
                                   'in_file',
                                   'another_file'])
    n_trigger = False
    # default value for "n"
    n = 1
    # default value for "verbose"
    verbose = 0
    # list to hold the positional arguments
    positional = []
    for arg in cmd_line:
        if n_trigger:
            n = int(arg)
            n_trigger = False
            continue
        if arg == '-n':
            # next argument belongs to "-n"
            n_trigger = True
        elif arg == '--verbose' or arg == '-v':
            verbose += 1
        else:
            # it must be the positional argument
            positional.append(arg)
    return Args(n=n,
                verbose=verbose,
                in_file=positional[0],
                another_file=positional[1])

# immaginary command line with increased verbosity
cmd_line = '-n 5 -v -v myfile.txt another_one.txt'
parse_args(cmd_line.split())

# by convention we can also increase verbosity in the following manner
cmd_line = '-n 5 -vvv myfile.txt another_one.txt'
parse_args(cmd_line)

def parse_args(cmd_line):
    Args = collections.namedtuple('Args',
                                  ['n',
                                   'verbose',
                                   'in_file',
                                   'another_file'])
    n_trigger = False
    # default value for "n"
    n = 1
    # default value for "verbose"
    verbose = 0
    # list to hold the positional arguments
    positional = []
    for arg in cmd_line:
        if n_trigger:
            n = int(arg)
            n_trigger = False
            continue
        if arg == '-n':
            # next argument belongs to "-n"
            n_trigger = True
        elif arg == '--verbose' or arg == '-v' or arg.startswith('-v'):
            if arg.startswith('-v') and len(arg) > 2 and len({char for char in arg[1:]}) == 1:
                verbose += len(arg[1:])
            else:
                verbose += 1
        else:
            # it must be the positional argument
            positional.append(arg)
    return Args(n=n,
                verbose=verbose,
                in_file=positional[0],
                another_file=positional[1])

# by convention we can also increase verbosity in the following manner
cmd_line = '-n 5 -vvv myfile.txt another_one.txt'
parse_args(cmd_line.split())

def parse_args(cmd_line):
    parser = argparse.ArgumentParser(prog='fake_script',
                                     description='An argparse test')
    
    # positional arguments
    parser.add_argument('my_file',
                        help='My input file')
    parser.add_argument('another_file',
                        help='Another input file')
    
    # optional arguments
    parser.add_argument('-n',
                        type=int,
                        default=1,
                        help='Number of Ns [Default: 1]')
    parser.add_argument('-v', '--verbose',
                        action='count',
                        default=0,
                        help='Increase verbosity level')
    
    return parser.parse_args(cmd_line)

# by convention we can also increase verbosity in the following manner
cmd_line = '-n 5 -vvv myfile.txt another_one.txt'
parse_args(cmd_line.split())

# by convention we can also increase verbosity in the following manner
cmd_line = '-n not_an_integer -vvv myfile.txt another_one.txt'
parse_args(cmd_line.split())

# by convention we can also increase verbosity in the following manner
cmd_line = '-h'
parse_args(cmd_line.split())

def parse_args(cmd_line):
    parser = argparse.ArgumentParser(prog='fake_script',
                                     description='An argparse example')
    
    # boolean option
    parser.add_argument('-f',
                        '--force',
                        action='store_true',
                        default=False,
                        help='Force file creation')
    
    return parser.parse_args(cmd_line)

cmd_line = '-f'
parse_args(cmd_line.split())

cmd_line = ''
parse_args(cmd_line.split())

def parse_args(cmd_line):
    parser = argparse.ArgumentParser(prog='fake_script',
                                     description='An argparse example')
    
    # multiple choices positional argument
    parser.add_argument('-m',
                        '--metric',
                        choices=['jaccard',
                                 'hamming'],
                        default='jaccard',
                        help='Distance metric [Default: jaccard]')
    parser.add_argument('-b',
                        '--bootstraps',
                        type=int,
                        choices=range(10, 21),
                        default=10,
                        help='Bootstraps [Default: 10]')
    
    return parser.parse_args(cmd_line)

cmd_line = '-m euclidean'
parse_args(cmd_line.split())

cmd_line = '-m hamming -b 15'
parse_args(cmd_line.split())

def parse_args(cmd_line):
    parser = argparse.ArgumentParser(prog='fake_script',
                                     description='An argparse example')
    
    parser.add_argument('fastq',
                        nargs='+',
                        help='Input fastq files')
    parser.add_argument('-m',
                        '--mate-pairs',
                        nargs='*',
                        help='Mate pairse fastq files')
    
    return parser.parse_args(cmd_line)

cmd_line = 'r1.fq.gz r2.fq.gz'
parse_args(cmd_line.split())

cmd_line = 'r1.fq.gz r2.fq.gz -m m1.fq.gz m2.fq.gz'
parse_args(cmd_line.split())

cmd_line = '-m m1.fq.gz m2.fq.gz'
parse_args(cmd_line.split())

cmd_line = '-h'
parse_args(cmd_line.split())

def init(options):
    print('Init the project')
    print(options.name, options.description)
    
def add(options):
    print('Add an entry')
    print(options.ID, options.name,
          options.description, options.color)

def parse_args(cmd_line):
    parser = argparse.ArgumentParser(prog='fake_script',
                                     description='An argparse example')
    
    subparsers = parser.add_subparsers()
    
    parser_init = subparsers.add_parser('init',
                            help='Initialize the project')
    parser_init.add_argument('-n',
                             '--name',
                             default='Project',
                             help='Project name')
    parser_init.add_argument('-d',
                             '--description',
                             default='My project',
                             help='Project description')
    parser_init.set_defaults(func=init)
    
    parser_add = subparsers.add_parser('add',
                            help='Add an entry')
    parser_add.add_argument('ID',
                            help='Entry ID')
    parser_add.add_argument('-n',
                            '--name',
                            default='',
                            help='Entry name')
    parser_add.add_argument('-d',
                            '--description',
                            default = '',
                            help='Entry description')
    parser_add.add_argument('-c',
                            '--color',
                            default='red',
                            help='Entry color')
    parser_add.set_defaults(func=add)
    
    return parser.parse_args(cmd_line)

cmd_line = '-h'
parse_args(cmd_line.split())

cmd_line = 'init -h'
parse_args(cmd_line.split())

cmd_line = 'add -h'
parse_args(cmd_line.split())

cmd_line = 'init -n my_project -d awesome'
options = parse_args(cmd_line.split())
options.func(options)

cmd_line = 'add test -n entry1 -d my_entry'
options = parse_args(cmd_line.split())
options.func(options)

sys.stderr.write('Running an immaginary analysis on the input genes\n')
# the result of our immaginary analysis
value = 400
# regular output of our immaginary script
print('\t'.join(['gene1', 'gene2', str(value)]))

user_provided_value = 'a'
try:
    # impossible
    parameter = int(user_provided_value)
except ValueError:
    sys.stderr.write('Invalid type provided\n')
    sys.exit(1)

# create the logger
logger = logging.getLogger('fake_script')

# set the verbosity level
logger.setLevel(logging.DEBUG)

# we want the log to be redirected
# to std. err.
ch = logging.StreamHandler()
# we want a rich output with additional information
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
                              '%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

# debug message, will be shown given the level we have set
logger.debug('test')

logger.setLevel(logging.WARNING)
# debug message, will not be shown given the level we have set
logger.debug('not-so-interesting debugging information')
# warning message, will be shown
logger.warning('this might break our script, but i\'m not sure')

#!/usr/bin/env python
'''Description here'''

import logging
import argparse

def get_options():
    description = ''
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('name',
                        help='Name')

    return parser.parse_args()


def set_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    options = get_options()
    
    logger = set_logging()

