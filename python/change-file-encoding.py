import os, re
from os.path import isdir, join, exists, abspath

raw_path = u'./corpus/raw-docs'
reader_codec = 'thai'
writer_codec = 'utf8'

get_ipython().run_cell_magic('time', '', "n = 0\nfor root, dirs, files in os.walk(raw_path):\n    for name in files:\n        n += 1\n        file_path = join(root, name)\n#         print file_path\n        with open(file_path, 'r') as f:\n            content = f.read().decode(reader_codec)\n        with open(file_path, 'w') as f:\n            f.write(content.encode(writer_codec))\nprint '%d files encoded.' %n")



