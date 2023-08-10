get_ipython().run_cell_magic('file', 'foo.py', '"""\nWhen this file is imported with `import foo`,\nonly `useful_func1()` and `useful_func()` are loaded, \nand the test code `assert ...` is ignored. However,\nwhen we run foo.py as a script `python foo.py`, then\nthe two assert statements are run.\nMost commonly, the code under `if __naem__ == \'__main__\':`\nconsists of simple examples or test cases for the functions\ndefined in the moule.\n"""\n\ndef useful_func1():\n    pass\n\ndef useful_fucn2():\n    pass\n\nif __name__ == \'__main__\':\n    assert(useful_func1() is None)\n    assert(useful_fucn2() is None)')

import pkg.foo as foo

foo.f1()

pkg.foo.f1()

get_ipython().system(' cat pkg/sub1/more_sub1_stuff.py')

from pkg.sub1.more_sub1_stuff import g3

g3()

get_ipython().system(' cat pkg/sub2/sub2_stuff.py')

from pkg.sub2.sub2_stuff import h2

h2()

get_ipython().system(' ls -R sta663')

get_ipython().system(' cat sta663/run_sta663.py')

get_ipython().system(' cat sta663/setup.py')

get_ipython().run_cell_magic('bash', '', '\ncd sta663\npython setup.py sdist\ncd -')

get_ipython().system(' ls -R sta663')

get_ipython().run_cell_magic('bash', '', '\ncp sta663/dist/sta663-1.0.tar.gz /tmp\ncd /tmp\ntar xzf sta663-1.0.tar.gz\ncd sta663-1.0\npython setup.py install')



