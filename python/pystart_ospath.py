import os

#get current work directory.
cwd = os.getcwd()
print(cwd)

os.listdir(cwd)

#os.remove()

os.system('ls -l -h')

#可以取代操作系统特定的路径分割符。
os.sep

#字符串给出当前平台使用的行终止符
os.linesep

#返回一个路径的目录名和文件名
#os.path.split('C:\\Python25\\abc.txt')
os.path.split('Python25/abc.txt')

#检验给出的路径是一个文件还是目录
print(os.path.isdir(os.getcwd()))
#True
print(os.path.isfile('a.txt'))
#False

#检验给出的路径是否存在
print(os.path.exists("~"))
print(os.path.exists(os.getcwd()))

#获得绝对路径
os.path.abspath(".")

#规范path字符串形式
os.path.normpath(cwd)

#获得文件大小，如果name是目录返回0L
os.path.getsize(".")

#分离文件名与扩展名
os.path.splitext("mydir/test.txt")

#连接目录与文件名或目录
afile = os.path.join(os.path.abspath("."),"test.txt")
print(afile)

#返回文件名
os.path.basename(afile)

#返回文件路径
os.path.dirname(afile)



