#这个程序用于可视化实验数据，并计算磁场的散度和旋度
import numpy as np
import matplotlib.pyplot as plt
a=np.load('experiment400.npy')
fig=plt.figure(figsize=(10, 5))
ax=fig.add_subplot(111,projection='3d') 
ax.scatter(a[:,3],a[:,4],a[:,5])
print(a[(a[:,0]==0) & (a[:,1]==0) & (a[:,2]==0),:])
plt.show()
x,y,z=np.meshgrid(np.arange(-80,81,40),np.arange(-80,81,40),np.arange(-80,81,40),indexing='ij')
Bx,By,Bz=np.zeros(x.shape),np.zeros(y.shape),np.zeros(z.shape)
for i in range(a.shape[0]):
    Bx[int((a[i,0]+80)/40),int((a[i,1]+80)/40),int((a[i,2]+80)/40)]=a[i,3]
    By[int((a[i,0]+80)/40),int((a[i,1]+80)/40),int((a[i,2]+80)/40)]=a[i,4]
    Bz[int((a[i,0]+80)/40),int((a[i,1]+80)/40),int((a[i,2]+80)/40)]=a[i,5]
fig=plt.figure(figsize=(10, 5)) 
ax1=fig.add_subplot(111,projection='3d')
# ax1.scatter(x,y,z,c='r',marker='o')
# ax1.quiver(x,y,z,Bx,By,Bz,length=10,normalize=True)
# plt.show()
x1=x[1:4,1:4,1:4].flatten()
y1=y[1:4,1:4,1:4].flatten()
z1=z[1:4,1:4,1:4].flatten()
Bx1=Bx[1:4,1:4,1:4].flatten()
By1=By[1:4,1:4,1:4].flatten()
Bz1=Bz[1:4,1:4,1:4].flatten()
# ax1.scatter(x1,y1,z1,c='r',marker='o')
# ax1.quiver(x1,y1,z1,Bx1,By1,Bz1,length=10,normalize=True)
# plt.show()
b=np.concatenate((x1[:,np.newaxis],y1[:,np.newaxis],z1[:,np.newaxis],Bx1[:,np.newaxis],By1[:,np.newaxis],Bz1[:,np.newaxis]),axis=1)
b1=b.reshape(-1,1,6)
# a=a[~((a==b1).any(0).all(1)),:]
ax1.scatter(a[:98,0],a[:98,1],a[:98,2])
ax1.quiver(a[:98,0],a[:98,1],a[:98,2],a[:98,3],a[:98,4],a[:98,5],length=10,normalize=True)
plt.show()
#np.save('experiment2.npy',np.concatenate((a,b),axis=0))
# Calculate the divergence of the vector field
divergence = np.gradient(Bx, axis=0) + np.gradient(By, axis=1) + np.gradient(Bz, axis=2)

# Calculate the curl of the vector field
curl_x = np.gradient(Bz, axis=1) - np.gradient(By, axis=2)
curl_y = np.gradient(Bx, axis=2) - np.gradient(Bz, axis=0)
curl_z = np.gradient(By, axis=0) - np.gradient(Bx, axis=1)

# Display the results
print("Divergence of the vector field:")
print(np.mean(np.abs(divergence))/np.mean(np.sqrt(Bx**2+By**2+Bz**2)))  

print("\nCurl of the vector field:")
print(np.mean(np.sqrt(curl_x**2+curl_y**2+curl_z**2))/np.mean(np.sqrt(Bx**2+By**2+Bz**2)))   