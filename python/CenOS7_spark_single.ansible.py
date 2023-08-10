print('working direory is:') 
get_ipython().system('pwd')
print()
print('Try to create ./sshkey directory if we do not have one')
get_ipython().system('[ -d ./sshkey ] || mkdir ./sshkey')
print()
print('Check current directory')
get_ipython().system('ls -F .')

get_ipython().system('chmod 600 ./sshkey/*')

get_ipython().system('ansible all -m ping')

get_ipython().system('ansible-playbook centos7_spark_local.yml')

get_ipython().system('rm ./sshkey/*')

