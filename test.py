import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from data import *

# 创建一个新的图形  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  

field = data_generation(
                        dx=3.42,
                        dy=3.62,
                        dz=3.42,
                        Iz=1,
                        Ix=1,
                        Iy=1,
                        N_sample=100,
                        L=0.8,
                        radius1=1,
                        radius2=1,
                        a=3.00,
                        b=3.62,
                        radius=1
                       )

N_val=10
L=field.L
y_test_np_grid = np.linspace(-L, L, N_val)
x_test_np_grid = np.linspace(-L, L, N_val)
z_test_np_grid = np.linspace(-L, L, N_val)
xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, sparse=False,indexing='ij')
temp_final = np.zeros(np.append(np.shape(xx),3))
temp_final1 = np.zeros(np.append(np.shape(xx),3))
xxravel = xx.ravel()
yyravel = yy.ravel()
zzravel = zz.ravel()
temp_final11=field.reccircB(np.expand_dims(xxravel,1), np.expand_dims(yyravel,1), np.expand_dims(zzravel,1))
for i in range(xx.size):
    idx=np.unravel_index(i, np.shape(xx))
    temp_final[idx]=temp_final11[i]
    temp_final1[idx]=field.HelmholtzB(xxravel[i], yyravel[i], zzravel[i])
temp_final2 = temp_final+temp_final1
# 通过画箭头来表示矢量  
for i in range(xx.size):  
    idx=np.unravel_index(i, np.shape(xx))
    ax.quiver(xxravel[i],yyravel[i],zzravel[i], *temp_final[idx]*0.5)  
grad_x = np.gradient(temp_final[..., 0], axis=0)
grad_y = np.gradient(temp_final[..., 1], axis=1)
grad_z = np.gradient(temp_final[..., 2], axis=2)
curl_x = np.gradient(temp_final[...,2], axis=1) - np.gradient(temp_final[...,1], axis=2)
curl_y = np.gradient(temp_final[...,0], axis=2) - np.gradient(temp_final[...,2], axis=0)
curl_z = np.gradient(temp_final[...,1], axis=0) - np.gradient(temp_final[...,0], axis=1)
grad_x1 = np.gradient(temp_final1[..., 0], axis=0)
grad_y1 = np.gradient(temp_final1[..., 1], axis=1)
grad_z1 = np.gradient(temp_final1[..., 2], axis=2)
print(np.mean(np.abs(grad_x+grad_y+grad_z)))
print(np.mean(np.abs(grad_x1+grad_y1+grad_z1)))
print(np.mean(np.sqrt(curl_x**2+curl_y**2+curl_z**2)))
print(np.mean(np.sqrt(np.sum(temp_final**2,axis=3))))
# 设置坐标轴标签  
ax.set_xlabel('X')  
ax.set_ylabel('Y')  
# 显示图形  
plt.show()
print(np.mean(temp_final,axis=(0,1,2)))
#CS = plt.contourf(xx[:,:,0],yy[:,:,0],temp_final2[:,0,:,0], cmap='jet')
#plt.colorbar(CS)
#plt.show()
