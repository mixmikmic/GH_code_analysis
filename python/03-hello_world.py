print("Hello world!") # Modify me and push <SHIFT> + <RETURN>

def hello():
    print('Hello world!')
import dis

dis.dis(hello)

get_ipython().system('cat hello_world.py')

get_ipython().system('./hello_world.py')

get_ipython().system('python hello_world.py')

get_ipython().system('python -c "print(\'Hello world!\')"')

