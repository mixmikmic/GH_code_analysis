get_ipython().system('hostname')

get_ipython().system('hostname -I')

host_name = get_ipython().getoutput('hostname')
host_name

type(host_name)

get_ipython().magic('pinfo host_name')

dir_list = get_ipython().getoutput('ls -al')
dir_list

print(dir_list.n)

