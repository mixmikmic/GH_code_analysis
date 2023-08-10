s = '.git/objects/8e/28241360c472576e8caa944253d4af368d9081'

import re
git_pattern = re.compile(r'''
    .*/                      # anything and a slash
    (
        ([0-9a-fA-F]{2})     # 2 hexadecimal digits
        /                    # separated by a slash
        ([0-9a-fA-F]{38})    # 38 hexadecimal digits
    )$''', flags=re.VERBOSE)

m = git_pattern.match(s)
m

m.group(0)

m.group(1)

m.group(2)

m.group(3)

help(re.compile)

s = '.git/objects/8e/28241360c472576e8caa944253d4af368d9081'

import re
git_pattern = re.compile(r'''
    .*/                      # anything and a slash
    (?P<hash_with_slash>
        (?P<hash_directory> [0-9a-fA-F]{2})     # 2 hexadecimal digits
        /                    # separated by a slash
        (?P<hash_filename> [0-9a-fA-F]{38})    # 38 hexadecimal digits
    )$''', flags=re.VERBOSE)

m = git_pattern.match(s)
m

m.group(2)

m.group('hash_directory')

m.group(3)

m.group('hash_filename')

hash = m.group('hash_directory') + m.group('hash_filename')
hash

hash = ''.join(map(m.group, ('hash_directory', 'hash_filename')))
hash

hash = ''.join(map(lambda s: m.group('hash_%s' % s), ('directory', 'filename')))
hash

m.group(1)

m.group('hash_with_slash')

hash = ''.join(c for c in m.group('hash_with_slash') if c != '/')
hash

s

s = '8e/28241360c472576e8caa944253d4af368d9081'
m = git_pattern.match(s)
m

repr(m)

