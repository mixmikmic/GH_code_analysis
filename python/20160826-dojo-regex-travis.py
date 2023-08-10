from __future__ import print_function
import re

pattern = re.compile(r"""
        (?P<any>any4?)                                       # "any"
                                                             #   association
        |                                                    # or
        (?P<object_eq>object ([\w-]+) eq (\d+))             # object
        alone
                                                             #   association
        |                                                    # or
        (?P<object_range>object ([a-z0-9A-Z-]+) range (\d+) (\d+)) # object range
                                                             #   association
        |                                                    # or
        (?P<object_group>object-group ([a-z0-9A-Z-]+))             # object group
                                                             #   association
        |                                                    # or
        (?P<object_alone>object ([[a-z0-9A-Z-]+))                   # object alone
                                                             #   association
""", re.VERBOSE)

s = '''    object-group jfi-ip-ranges object DA-TD-WEB01 eq 8850
'''

pattern = re.compile(r"""
        (?P<any>any4?)                                       # "any"
                                                             #   association
        |                                                    # or
        (?P<object_eq>object\ ([\w-]+)\ eq\ (\d+))             # object
        alone
                                                             #   association
        |                                                    # or
        (?P<object_range>object\ ([a-z0-9A-Z-]+)\ range\ (\d+)\ (\d+)) # object range
                                                             #   association
        |                                                    # or
        (?P<object_group>object-group\ ([a-z0-9A-Z-]+))             # object group
                                                             #   association
        |                                                    # or
        (?P<object_alone>object\ ([a-z0-9A-Z-]+))                   # object alone
                                                             #   association
""", re.VERBOSE)

re.findall(pattern, s)

for m in re.finditer(pattern, s):
    print(repr(m))
    print('groups', m.groups())
    print('groupdict', m.groupdict())

pattern = re.compile(r"""
        (?P<any>any4?)                                       # "any"
                                                             #   association
        |                                                    # or
        (?P<object_eq>object\ (?P<oe_name>[\w-]+)\ eq\ (?P<oe_i>\d+))             # object
        alone
                                                             #   association
        |                                                    # or
        (?P<object_range>object\ (?P<or_name>[a-z0-9A-Z-]+)
        \ range\ (?P<oe_r_start>\d+)\ (?P<oe_r_end>\d+)) # object range
                                                             #   association
        |                                                    # or
        (?P<object_group>object-group\ (?P<og_name>[a-z0-9A-Z-]+))             # object group
                                                             #   association
        |                                                    # or
        (?P<object_alone>object\ (?P<oa_name>[a-z0-9A-Z-]+))                   # object alone
                                                             #   association
""", re.VERBOSE)

for m in re.finditer(pattern, s):
    print(repr(m))
    print('groups', m.groups())
    print('groupdict', m.groupdict())

for m in re.finditer(pattern, s):
    for key, value in m.groupdict().items():
        if value is not None:
            print(key, repr(value))
    print()

pattern = re.compile(r"""
        (?P<any>any4?)                                       # "any"
                                                             #   association
        |                                                    # or
        (object\ (?P<oe_name>[\w-]+)\ eq\ (?P<oe_i>\d+))             # object
        alone
                                                             #   association
        |                                                    # or
        (object\ (?P<or_name>[a-z0-9A-Z-]+)
        \ range\ (?P<oe_r_start>\d+)\ (?P<oe_r_end>\d+)) # object range
                                                             #   association
        |                                                    # or
        (object-group\ (?P<og_name>[a-z0-9A-Z-]+))             # object group
                                                             #   association
        |                                                    # or
        (object\ (?P<oa_name>[a-z0-9A-Z-]+))                   # object alone
                                                             #   association
""", re.VERBOSE)

for m in re.finditer(pattern, s):
    for key, value in m.groupdict().items():
        if value is not None:
            print(key, repr(value))
    print()

