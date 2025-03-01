import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from data import *
import torch
def getdB(Bpred, Breal):
    Bx = Bpred[:,0]
    By = Bpred[:,1]
    Bz = Bpred[:,2]
    Bx_r = Breal[:,0]
    By_r = Breal[:,1]
    Bz_r = Breal[:,2]
    dBx = (Bx - Bx_r)/Bx_r
    dBy = (By - By_r)/By_r
    dBz = (Bz - Bz_r)/Bz_r
    dB = ( np.sqrt(Bx**2+By**2+Bz**2) - np.sqrt(Bx_r**2+By_r**2+Bz_r**2) ) / np.sqrt(Bx_r**2+By_r**2+Bz_r**2)
    dBx[abs(dBx-np.mean(dBx))/np.std(dBx)>3]=0
    dBy[abs(dBy-np.mean(dBy))/np.std(dBy)>3]=0
    dBz[abs(dBz-np.mean(dBz))/np.std(dBz)>3]=0
    dB[abs(dB-np.mean(dB))/np.std(dB)>3]=0
    dBx[abs(Bx_r)<np.mean(abs(Bx_r))/3]=0
    dBy[abs(By_r)<np.mean(abs(By_r))/3]=0
    dBz[abs(Bz_r)<np.mean(abs(Bz_r))/3]=0
    dB[abs(np.sqrt(Bx_r**2+By_r**2+Bz_r**2))<np.mean(abs(np.sqrt(Bx_r**2+By_r**2+Bz_r**2)))/3]=0
    return dBx, dBy, dBz, dB
field = data_generation(
                        dx=0.358,
                        dy=0.316,
                        dz=0.358,
                        Iz=1,
                        Ix=1,
                        Iy=1,
                        N_sample=100,
                        L=0.15,
                        radius1=0.220,
                        radius2=0.207,
                        a=0.272,
                        b=0.378,
                       )
data = np.loadtxt('B(1,1,1)2.fld',skiprows=2)
N_val=31
L=field.L
y_test_np_grid = np.linspace(-L, L, N_val)
x_test_np_grid = np.linspace(-L, L, N_val)
z_test_np_grid = np.linspace(-L, L, N_val)
xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, indexing='ij')
temp_final = np.zeros(np.append(np.shape(xx),3))
xxravel = xx.ravel()
yyravel = yy.ravel()
zzravel = zz.ravel()
temp_final11=field.reccircB(np.expand_dims(xxravel,1), np.expand_dims(yyravel,1), np.expand_dims(zzravel,1))
temp_final11 = temp_final11/np.std(temp_final11,axis=0)
temp_final=np.reshape(temp_final11, np.append(np.shape(xx),3))
    

#fig=plt.figure()
#fig1=plt.figure()
#ax = fig.add_subplot(111, projection='3d') 
#ax1 = fig1.add_subplot(111, projection='3d')

data1=data[:,3:6]/np.std(data[:,3:6],axis=0)
#ax.quiver(xxravel[::10],yyravel[::10],zzravel[::10], temp_final11[::10,0]*0.05,temp_final11[::10,1]*0.05,temp_final11[::10,2]*0.05)
#ax1.quiver(xxravel[::10],yyravel[::10],zzravel[::10], data1[::10,0]*0.05,data1[::10,1]*0.05,data1[::10,2]*0.05)
#plt.show()


coord=np.concatenate((np.expand_dims(xxravel,1),np.expand_dims(yyravel,1),np.expand_dims(zzravel,1)),axis=1)
#print(np.mean(np.abs(temp_final11-data1))/np.mean(np.abs(data1)))
print(np.mean(np.abs(coord-data[:,0:3])))
print(np.std(data1,axis=0),np.std(temp_final11,axis=0))
#print(np.mean(data1,axis=0))
#print(temp_final11-data1)

print(temp_final11.shape,data1.shape)
dBx, dBy, dBz, dB = getdB(temp_final11,data1,)
fig_stat = plt.figure(figsize=([16,16]))
fig_stat.add_subplot(2,2,1)
plt.hist(dBx, bins=10, label=f"Bx_pred - Bx_real: mean {dBx.mean():.5f} std {dBx.std():.5f}")
plt.legend()
plt.yscale('log')
fig_stat.add_subplot(2,2,2)
plt.hist(dBy, bins=10, label=f"By_pred - By_real: mean {dBy.mean():.5f} std {dBy.std():.5f}")
plt.legend()
plt.yscale('log')
fig_stat.add_subplot(2,2,3)
plt.hist(dBz, bins=10, label=f"Bz_pred - Bz_real: mean {dBz.mean():.5f} std {dBz.std():.5f}")
plt.legend()
plt.yscale('log')
fig_stat.add_subplot(2,2,4)
plt.hist(dB, bins=10, label=f"B_pred - B_real: mean {dB.mean():.5f} std {dB.std():.5f}")
plt.legend()
plt.yscale('log')
plt.show()
plt.close()



model_output = np.reshape(data1,(N_val, N_val, N_val, 3))

ax = ['Bx', 'By', 'Bz']
idxlist=np.linspace(0,N_val-1,5,dtype=int)
#在不同的z方向上看xy平面的结果
for i in range(3):
    figure = plt.figure(figsize=(20,30))
    real = model_output[:,:,:,i]
    pred = temp_final[:,:,:,i]
    for j in range(5):
        idx=idxlist[j]
        pred_slice = pred[:,:,idx]                       
        real_slice = real[:,:,idx]
        figure.add_subplot(5,3,j*3+1)
        #plt.rc('font', size=16)
        X = xx[:,:,idx]
        Y = yy[:,:,idx]
        CS = plt.contourf(X,Y, pred_slice, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'pred filed at z={np.round(-L+idx*L*2/N_val,2)}') 
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)   
        figure.add_subplot(5,3,2+3*j)
        #plt.rc('font', size=16)            
        CS = plt.contourf(X, Y, real_slice, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'truth field at z={np.round(-L+idx*L*2/N_val,2)}')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        figure.add_subplot(5,3,3*j+3)
        #plt.rc('font', size=16)
        a=(pred_slice-real_slice)/real_slice
        a[abs(a-np.mean(a))/np.std(a)>3]=0
        a[abs(real_slice)<np.mean(abs(real_slice))/3]=0
        CS = plt.contourf(X,Y,a, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('relative error pred-truth/truth')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    plt.close()

#在不同的y方向上看zx平面的结果
for i in range(3):
    figure = plt.figure(figsize=(20,30))
    real = model_output[:,:,:,i]
    pred = temp_final[:,:,:,i]
    for j in range(5):
        idx=idxlist[j]
        pred_slice = pred[idx,:,:]                       
        real_slice = real[idx,:,:]
        figure.add_subplot(5,3,j*3+1)
        #plt.rc('font', size=16)
        X = xx[:,idx,:]
        Z = zz[:,idx,:]
        CS = plt.contourf(X,Z, pred_slice, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'pred filed at y={np.round(-L+idx*L*2/N_val,2)}')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)   
        figure.add_subplot(5,3,j*3+2)
        #plt.rc('font', size=16)
        CS = plt.contourf(X,Z, real_slice, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'truth field at y={np.round(-L+idx*L*2/N_val,2)}')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)   
        figure.add_subplot(5,3,j*3+3)
        #plt.rc('font', size=16)
        a=(pred_slice-real_slice)/real_slice
        a[abs(a-np.mean(a))/np.std(a)>3]=0
        a[abs(real_slice)<np.mean(abs(real_slice))/3]=0
        CS = plt.contourf(X,Z,a, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('relative error pred-truth/truth')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)   
        plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    plt.close()            

#在不同的x方向上看yz平面的结果
for i in range(3):
    figure = plt.figure(figsize=(20,30))
    real = model_output[:,:,:,i]
    pred = temp_final[:,:,:,i]
    for j in range(5):
        idx=idxlist[j]
        pred_slice = pred[:,idx,:]
        real_slice = real[:,idx,:]
        figure.add_subplot(5,3,j*3+1)
        #plt.rc('font', size=16)
        Y = yy[idx,:,:]
        Z = zz[idx,:,:]
        CS = plt.contourf(Y,Z, pred_slice, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title(f'pred filed at x={np.round(-L+idx*L*2/N_val,decimals=2)}')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)   
        figure.add_subplot(5,3,j*3+2)
        #plt.rc('font', size=16)
        CS = plt.contourf(Y, Z, real_slice, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title(f'truth field at x={np.round(-L+idx*L*2/N_val,decimals=2)}')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)   
        plt.gca().axes.get_yaxis().set_visible(False)
        figure.add_subplot(5,3,j*3+3)
        #plt.rc('font', size=16)
        a=(pred_slice-real_slice)/real_slice
        a[abs(a-np.mean(a))/np.std(a)>3]=0
        a[abs(real_slice)<np.mean(abs(real_slice))/3]=0
        CS = plt.contourf(Y,Z,a, N_val, cmap='jet')
        plt.colorbar(CS)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('relative error pred-truth/truth')
        if(not j==4):
            plt.gca().axes.get_xaxis().set_visible(False)   
        plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    plt.close()