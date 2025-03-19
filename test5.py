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
# 提取Bx, By, Bz列
Bx = data['Bx'].values
By = data['By'].values
Bz = data['Bz'].values

# 画出Bx, By, Bz随时间变化的图
plt.figure(figsize=(10, 6))
plt.plot(data['时间'], Bx, label='Bx')
#plt.plot(data['时间'], By, label='By')
#plt.plot(data['时间'], Bz, label='Bz')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Magnetic Field Intensity')
plt.title('Time vs Magnetic Field Intensity')
window=5/((data['时间'].iloc[-1]-data['时间'].iloc[0]).total_seconds()/data.shape[0])
window=int(np.round(window))
# 挑出16:13:25.000之前的数据
data = data[data['时间'] < pd.to_datetime('16:13:25.000', format='%H:%M:%S.%f')]
B=np.zeros((data.shape[0]-window+1,3))
Bstd=np.zeros((data.shape[0]-window+1))
time = pd.DataFrame(index=range(data.shape[0]-window+1), columns=['时间'])
for i in range(data.shape[0]-window+1):
    B[i,:]=np.mean(data[['Bx','By','Bz']].iloc[i:i+window],axis=0)
    Bstd[i]=np.sum((np.std(data[['Bx','By','Bz']].iloc[i:i+window],axis=0))**2)  
    time.iloc[i] = data['时间'].iloc[i:i+window].mean()
theshold=0.07**2
B=B[Bstd<theshold,:]
time=time[Bstd<theshold]
Bstd=Bstd[Bstd<theshold]
time['时间']=pd.to_datetime(time['时间'], format='%H:%M:%S.%f')
# 合并B中连续的相似值
merged_B = []
merged_time = pd.DataFrame(columns=['时间'])
merged_Bstd = []

i = 0
while i < len(B):
    start = i
    while i + 1 < len(B) and np.allclose(B[i], B[i + 1], atol=np.sqrt(theshold)+0.57):
        i += 1
    merged_B.append(np.mean(B[start:i + 1], axis=0))
    merged_time=pd.concat([merged_time,pd.DataFrame({'时间':time.iloc[start:i+1].mean()})])
    merged_Bstd.append(np.mean(Bstd[start:i + 1]))
    i += 1

B = np.array(merged_B)
Bstd = np.array(merged_Bstd)
time = merged_time
# 画出Bx, By, Bz随时间变化的图
print(B.shape[0])
plt.scatter(time,B[:,0],label='Bx')
#plt.plot(time,B[:,1],label='By') 
#plt.plot(time,B[:,2],label='Bz')
plt.show()

# 计算磁场数据随时间的梯度
# data['时间差'] = data['时间'].diff().dt.total_seconds()
# data['Bx梯度'] = data['Bx'].diff() / data['时间差']
# data['By梯度'] = data['By'].diff() / data['时间差']
# data['Bz梯度'] = data['Bz'].diff() / data['时间差']
# data['总梯度'] = np.sqrt(data['Bx梯度']**2 + data['By梯度']**2 + data['Bz梯度']**2)

# 去除第一个NaN值
# data[0,'总梯度'] = 0
# 以时间为横轴，B梯度为纵轴画图
# plt.figure(figsize=(10, 6))
# plt.scatter(data['时间'], data['总梯度']>0.209) 
# plt.xlabel('Time')
# plt.ylabel('B Gradient')
# plt.title('Time vs B Gradient with Clusters')
# plt.show()
# Bx=0
# By=0
# Bz=0
# j=0
# Bxlist=np.array([])
# Bylist=np.array([])
# Bzlist=np.array([])
# for i in range(data.shape[0]):
#     if data['总梯度'][i] < 0.20813 and j<40:
#         Bx+=data['Bx'][i]
#         By+=data['By'][i]
#         Bz+=data['Bz'][i]
#         j+=1
#     elif(j>10):
#         Bxlist=np.append(Bxlist,Bx/j)
#         Bylist=np.append(Bylist,By/j)
#         Bzlist=np.append(Bzlist,Bz/j)
#         Bx=0
#         By=0
#         Bz=0
#         j=0
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Bxlist, Bylist, Bzlist, c='r', marker='o')
# ax.set_xlabel('Bx')
# ax.set_ylabel('By')
# ax.set_zlabel('Bz')
# plt.title('3D Scatter Plot of Bx, By, Bz')
# plt.show()
# B=np.concatenate((Bxlist,Bylist,Bzlist),axis=0)
# B=np.reshape(B,(-1,3),order='F')
coords=[0,0,40,0,40,40,0,40,-40,40,-40,0,-40,-40,0,-40,40,-40,80,-40,80,0,80,40,80,80,40,80,0,80,-40,80,-80,80,-80,40,-80,0,-80,-40,-80,-80,-40,-80,0,-80,40,-80,80,-80]
coords=np.reshape(coords,(-1,2))
coords1=[0,40,80,-40,-80]
for i in range(len(coords1)):
    if(i==0):
        coords2=np.concatenate((np.repeat([coords1[i]],coords.shape[0])[:,np.newaxis],coords),axis=1)
    else:
        coords2=np.append(coords2,np.concatenate((np.repeat(coords1[i],coords.shape[0])[:,np.newaxis],coords),axis=1),axis=0)
np.save('expe_new.npy',np.concatenate((coords2,B),axis=1))   
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c='b', marker='o')

# for i in range(coords2.shape[0]):
#     ax.text(coords2[i, 0], coords2[i, 1], coords2[i, 2], '%d' % i, size=10, zorder=1, color='k')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Scatter Plot of Coords with Labels')
# plt.show()