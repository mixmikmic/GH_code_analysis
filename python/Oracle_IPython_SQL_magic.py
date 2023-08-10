# loads the SQL magic extensions
get_ipython().magic('load_ext sql')

# Connect to Oracle
get_ipython().magic('sql oracle+cx_oracle://scott:tiger@dbserver:1521/?service_name=orcl.mydomain.com')

get_ipython().run_cell_magic('sql', '', 'select * from emp')

Employee_name="SCOTT"

get_ipython().magic('sql select * from emp where ename=:Employee_name')

get_ipython().magic('sql update emp set sal=3500 where ename=:Employee_name')
get_ipython().magic('sql commit')
get_ipython().magic('sql select * from emp where ename=:Employee_name')

myResultSet = get_ipython().magic('sql select ename "Employee Name", sal "Salary" from emp')

get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')

myResultSet.bar()

get_ipython().run_cell_magic('sql', '', 'select e1.ename "Employee Name", e1.job "Job", e2.ename "Manager Name" \nfrom emp e1, emp e2\nwhere e1.mgr = e2.empno(+)')

# save result set into my_ResultSet and copy it to pandas in my_DataFrame
my_ResultSet = _

my_DataFrame=my_ResultSet.DataFrame()

my_DataFrame.head()



