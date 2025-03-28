import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from data import *
import torch
from scipy.optimize import minimize
data=np.load('experiment2.npy')
data[:,0:3]=data[:,0:3]/1000
B_real=data[:,3:6]
#B_real=B_real/np.std(B_real,axis=0)
def objective(x):
    field = data_generation(
                            dx=0.358,
                            dy=0.316,
                            dz=0.358,
                            Iz=x[2],
                            Ix=x[0],
                            Iy=x[1],
                            N_sample=100,
                            L=0.15,
                            radius1=0.220,
                            radius2=0.207,
                            a=0.272,
                            b=0.378,
                        )
    theta=np.pi/2
    phi=np.pi/2
    x=data[:,0]
    y=data[:,1]
    z=data[:,2]
    x_prime=x*np.cos(theta)*np.cos(phi)+y*np.cos(theta)*np.sin(phi)-z*np.sin(theta)
    y_prime=-x*np.sin(phi)+y*np.cos(phi)
    z_prime=x*np.sin(theta)*np.cos(phi)+y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    x_prime=np.append(x_prime,0)
    y_prime=np.append(y_prime,0)
    z_prime=np.append(z_prime,0)
    B_pred=np.array(field.reccircB(np.expand_dims(x_prime,1), np.expand_dims(y_prime,1), np.expand_dims(z_prime,1)))
    Bx_prime=B_pred[:,0]
    By_prime=B_pred[:,1]
    Bz_prime=B_pred[:,2]
    Bx=Bx_prime*np.cos(theta)*np.cos(phi)-By_prime*np.sin(phi)+Bz_prime*np.sin(theta)*np.cos(phi)
    By=Bx_prime*np.cos(theta)*np.sin(phi)+By_prime*np.cos(phi)+Bz_prime*np.sin(theta)*np.sin(phi)
    Bz=-Bx_prime*np.sin(theta)+Bz_prime*np.cos(theta)
    B_pred=np.column_stack((Bx,By,Bz))
    #B_pred=B_pred/np.std(B_pred,axis=0)
    B_pred=B_pred[:-1,:]-B_pred[-1,:]
    return np.mean((B_pred-B_real)**2)
x0=[1000,1000,0]
res = minimize(objective, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x/res.x[2])