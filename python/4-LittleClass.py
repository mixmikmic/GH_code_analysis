get_ipython().magic('reload_ext pytriqs.magic')

get_ipython().run_cell_magic('triqs', '', '#include <vector>\n         \n/**\n   Doc of the class \n*/    \nclass a_c_class { \n int x;\n    \n public:\n \n /// Doc of the constructor    \n a_c_class(int i) : x(i) {}\n    \n /// DOcumentation of get_my x   \n int getmyx() const { return x;}\n    \n};        ')

a = ACClass(1)
print a.getmyx

help(ACClass)



