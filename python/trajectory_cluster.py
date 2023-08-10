'''
预定义两个阈值：探测半径和相似度阈值
'''
detect_radius = 40
similarity_threashold = 0.5

# 数据读取和转换成标准numpy数组
file = open("TrackData-3.txt")
import re

P = []
for path in file.readlines():
    path = re.sub('Track-\d+:\d+:','',path)
    path=path.strip('\n')
    path = path.split('#')
    nPath = []
#     print path
    for point in path:
        point = [x,y] = point.split(',')
        nPath.append(point)
#     print(nPath)
    X = []
    Y = []
    for point in nPath:
        X.append(float(point[0]))
        Y.append(float(point[1]))
#     print X,Y
    P.append([X,Y])

from matplotlib import pyplot as plt
colors = ['r','g','y','b','k','c','m']

rect = [0, 0, 5, 5]
ax = plt.axes(rect)
for index,path in enumerate(P):
    ax.plot(path[0],path[1],colors[index%7])
plt.show()

import math
# 判断两个点是否是邻近点
def is_closed(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < detect_radius

# 计算r1对r2的相似度，注意这个地方的r1->r2的相似度和r2->r1的相似度不相同
def similarity(r1, r2):
    t = [0 for x in r1[0]]
    j = 0
    for idx1,x1 in enumerate(r1[0]):
        for idx2,x2 in enumerate(r2[0]): 
            if(idx1 < len(r1[0]) and idx2 < len(r2[0])):
                if is_closed([x1,r1[1][idx1]],[x2,r2[1][idx2]]):
                    t[idx1] = 1
                    break
                else:
                    j = j + 1
    s = [x for x in t if x ==1] 
    return float(len(s))/float(len(t))

# 计算相似度矩阵
import numpy as np
simi = np.zeros((len(P),len(P)))
for i1, r1 in enumerate(P):
    for i2, r2 in enumerate(P):
            simi[i1][i2] = similarity(r1, r2)

print(simi)

# 对轨迹进行聚类
clusters = [0] * len(simi)
i = 1
for x, row in enumerate(simi):
    for y, item in enumerate(row):
        if(item >= similarity_threashold and simi[y][x] >= similarity_threashold):
            if(clusters[x] != 0):
                clusters[y] = clusters[x]
            else:
                clusters[y] = clusters[x] = i
                i = i + 1 
i = i - 1
print(clusters)
print 'Total cluster is:', i 

ax = plt.axes(rect)
j = 1
while j <= i:
    k = 0
    while(k < len(clusters)):
        if clusters[k] == j:
            ax.plot(P[k][0],P[k][1],colors[j-1])
        k = k + 1
    j = j + 1

# ax.plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5)
plt.show()



