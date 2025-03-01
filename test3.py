import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Load arm_pos.csv into a numpy array
arm_pos=pd.read_csv('arm_pos.csv')

data_sensor=pd.read_csv('data_sensor.csv')



#print("Arm Position Data:")
#print(arm_pos)
#print("\nSensor Data:")
#print(data_sensor)

a0=np.array(data_sensor.iloc[:,1:4].values)
b0=np.array(arm_pos.iloc[:,2:8].values)
idx=np.array(arm_pos.iloc[:,0].values)
t=arm_pos.time
#b=np.zeros((a.shape[0],6))
#for i in range(a.shape[0]):
    #b[i]=b0[pd.Series(abs(pd.to_datetime(arm_pos.time,format='%H:%M:%S')-pd.to_datetime(data_sensor.index[i],format=' %H:%M:%S.%f'))).idxmin(),:]
    #print(i/a.shape[0]*100)
unique,indices,counts=np.unique(b0[:,0:3],axis=0,return_index=True,return_counts=True)

indices=indices[np.argsort(counts)[::-1]]
unique=unique[np.argsort(counts)[::-1],:]
counts=np.sort(counts)[::-1]
arm_pos_final=np.zeros((max(idx)+1,3))
data_sensor_final=np.zeros((max(idx)+1,3))
idxlist=np.array([])
i=0
while (np.size(idxlist)<max(idx)+1):
    a = np.all(b0[:, :3] == unique[i, :], axis=1)
    idx_temp=idx[a]
    b0_temp=b0[a,:]
    b0_temp=b0_temp[idx_temp==np.rint(np.mean(idx_temp)),:]   
    b0_temp=b0_temp[int(np.rint(np.size(b0_temp,axis=0)/2)):,:] 
    if(np.sum(idxlist==np.rint(np.mean(idx_temp)))):
        i+=1
        continue
    idxlist=np.append(idxlist,np.rint(np.mean(idx_temp)))
    t_temp=t[a]
    t_temp=t_temp[idx_temp==np.rint(np.mean(idx_temp))]   
    b=np.zeros((b0_temp.shape[0],3))
    t_temp=t_temp.reset_index(drop=True)
    for j in range(b0_temp.shape[0]):
        b[j]=a0[pd.Series(abs(pd.to_datetime(t_temp[j],format='%H:%M:%S')-pd.to_datetime(data_sensor.index,format=' %H:%M:%S.%f'))).idxmin(),:]
    print(np.mean(idx_temp))
    data_sensor_final[np.size(idxlist)-1,:]=np.mean(b,axis=0)
    arm_pos_final[np.size(idxlist)-1,:]=np.mean(b0_temp[:,0:3],axis=0)
    i+=1
    print(np.size(idxlist)/(max(idx)+1)*100)
print(arm_pos_final.shape,data_sensor_final.shape)
i=0
while i<np.size(arm_pos_final,axis=0):
    r=np.sum((arm_pos_final-arm_pos_final[i,:])**2,axis=1)
    a=r<100
    arm_pos_final[i,:]=np.mean(arm_pos_final[a,:],axis=0)
    data_sensor_final[i,:]=np.mean(data_sensor_final[a,:],axis=0)
    a[i]=False
    idxlist = np.delete(idxlist, np.where(a))
    arm_pos_final =np.delete(arm_pos_final, np.where(a),axis=0)
    data_sensor_final =np.delete(data_sensor_final, np.where(a),axis=0)
    i+=1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(arm_pos_final[:, 0], arm_pos_final[:, 1], arm_pos_final[:, 2], c='r', marker='o')
for i, txt in enumerate(idxlist):
    ax.text(arm_pos_final[i, 0], arm_pos_final[i, 1], arm_pos_final[i, 2], '%d' % txt, size=10, zorder=1, color='k')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

plt.show()
        
np.save('experiment1',np.concatenate((arm_pos_final,data_sensor_final),axis=1))

    