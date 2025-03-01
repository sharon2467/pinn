import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('data.csv',sep='\t')

# 假设数据有三列：时间、位置、磁场强度
# 进行聚类操作

# 确保以pandas读取时间数据的方式来读第一列
data['时间'] = pd.to_datetime(data['时间'].str.strip(), format='%H:%M:%S.%f')
# 计算磁场数据随时间的梯度
data['时间差'] = data['时间'].diff().dt.total_seconds()
data['Bx梯度'] = data['Bx'].diff() / data['时间差']
data['By梯度'] = data['By'].diff() / data['时间差']
data['Bz梯度'] = data['Bz'].diff() / data['时间差']
data['总梯度'] = np.sqrt(data['Bx梯度']**2 + data['By梯度']**2 + data['Bz梯度']**2)

# 去除第一个NaN值
data[0,'总梯度'] = 0
# 以时间为横轴，B梯度为纵轴画图
plt.figure(figsize=(10, 6))
plt.scatter(data['时间'], data['总梯度']>0.209) 
plt.xlabel('Time')
plt.ylabel('B Gradient')
plt.title('Time vs B Gradient with Clusters')
plt.show()
Bx=0
By=0
Bz=0
j=0
Bxlist=np.array([])
Bylist=np.array([])
Bzlist=np.array([])
for i in range(data.shape[0]):
    if data['总梯度'][i] < 0.20813 and j<40:
        Bx+=data['Bx'][i]
        By+=data['By'][i]
        Bz+=data['Bz'][i]
        j+=1
    elif(j>10):
        Bxlist=np.append(Bxlist,Bx/j)
        Bylist=np.append(Bylist,By/j)
        Bzlist=np.append(Bzlist,Bz/j)
        Bx=0
        By=0
        Bz=0
        j=0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Bxlist, Bylist, Bzlist, c='r', marker='o')
ax.set_xlabel('Bx')
ax.set_ylabel('By')
ax.set_zlabel('Bz')
plt.title('3D Scatter Plot of Bx, By, Bz')
plt.show()
B=np.concatenate((Bxlist,Bylist,Bzlist),axis=0)
B=np.reshape(B,(-1,3),order='F')
coords=[0,0,40,0,40,40,0,40,-40,40,-40,0,-40,-40,0,-40,40,-40,80,-40,80,0,80,40,80,80,40,80,0,80,-40,80,-80,80,-80,40,-80,0,-80,-40,-80,-80,-40,-80,0,-80,40,-80,80,-80]
coords=np.reshape(coords,(-1,2))
coords1=[0,40,80,-40,-80]
for i in range(len(coords1)):
    if(i==0):
        coords2=np.concatenate((np.repeat([coords1[i]],coords.shape[0])[:,np.newaxis],coords),axis=1)
    else:
        coords2=np.append(coords2,np.concatenate((np.repeat(coords1[i],coords.shape[0])[:,np.newaxis],coords),axis=1),axis=0)
np.save('expe_new.npy',np.concatenate((coords2,B[:-1,:]),axis=1))   
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c='b', marker='o')

for i in range(coords2.shape[0]):
    ax.text(coords2[i, 0], coords2[i, 1], coords2[i, 2], '%d' % i, size=10, zorder=1, color='k')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot of Coords with Labels')
plt.show()