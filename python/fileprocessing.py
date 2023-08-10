import os  
allFileNum = 0  
gen = 0;
def printPath(level, path): 
    global allFileNum  
    global gen
    ''''' 
    打印一个目录下的所有文件夹和文件 
    '''  
    # 所有文件夹，第一个字段是次目录的级别  
    dirList = []  
    # 所有文件  
    fileList = []  
    # 返回一个列表，其中包含在目录条目的名称(google翻译)  
    files = os.listdir(path)  
    # 先添加目录级别  
    dirList.append(str(level))  
    for f in files:  
        if(os.path.isdir(path + '/' + f)):  
            # 排除隐藏文件夹。因为隐藏文件夹过多  
            if(f[0] == '.'):  
                pass  
            else:  
                # 添加非隐藏文件夹  
                dirList.append(f)  
        if(os.path.isfile(path + '/' + f)):
            if(f[0] == '.'):  
                pass  
            else:
            # 添加文件  
                fileList.append(f)  
    # 当一个标志使用，文件夹列表第一个级别不打印  
    i_dl = 0  
    for dl in dirList:  
        if(i_dl == 0):  
            i_dl = i_dl + 1  
        else:  
            # 打印至控制台，不是第一个的目录  
            #print '-' * (int(dirList[0])), dl  
            # 打印目录下的所有文件夹和文件，目录级别+1  
            
            printPath((int(dirList[0]) + 1), path + '/' + dl)
    #print path        
    #gen = gen + len(fileList) * (len(fileList)+1)/2 - len(fileList)
    for i in xrange(len(fileList)):
            print path + '/' + fileList[i]
        
    #for fl in fileList:  
        # 打印文件  
        #print path + '/' + fl, len(fileList) * (len(fileList))
        # 随便计算一下有多少个文件  
        #allFileNum = allFileNum + 1  


filePath = "n4CASIA"
printPath(0,filePath)
print gen

for i in xrange(3,10):
    print i

for a in range(5,10):
    print a
    

with open('files.txt') as f:
    a = f.readlines();
print len(a)
a[-1] = a[-1]+'\n'

with open('imposter.txt','wr') as f:
    for i in xrange(5237):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i+1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter1.txt','wr') as f:
    for i in xrange(500):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i+1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')
        

with open('imposter2.txt','wr') as f:
    for i in xrange(500,1000):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')
        

with open('imposter3.txt','wr') as f:
    for i in xrange(1000,1500):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter4.txt','wr') as f:
    for i in xrange(1500,2000):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter5.txt','wr') as f:
    for i in xrange(2000,2500):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter6.txt','wr') as f:
    for i in xrange(2500,3000):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter7.txt','wr') as f:
    for i in xrange(3000,3500):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter8.txt','wr') as f:
    for i in xrange(3500,4000):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter9.txt','wr') as f:
    for i in xrange(4000,4500):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter10.txt','wr') as f:
    for i in xrange(4500,5237):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

with open('imposter2.txt','wr') as f:
    for i in xrange(500,1000):
        gen = a[i];
        genstart = gen[0:13]
        for j in xrange(i + 1,5237):
            if a[j].startswith(genstart):
                continue
            else:
                f.write(a[i][0:-1]+' '+a[j][0:-1]+'\n')

