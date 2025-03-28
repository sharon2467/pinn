import torch
from data import *
from model import *
import matplotlib.pyplot as plt
def getdB(Bpred, Breal):
    Bx = Bpred[:,0].detach().numpy()
    By = Bpred[:,1].detach().numpy()
    Bz = Bpred[:,2].detach().numpy()
    Bx_r = Breal[:,0]
    By_r = Breal[:,1]
    Bz_r = Breal[:,2]
    Bx_r[Bx_r==0]=Bx[Bx_r==0]
    By_r[By_r==0]=By[By_r==0]
    Bz_r[Bz_r==0]=Bz[Bz_r==0]
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
    dBx=dBx[dBx!=0]
    dBy=dBy[dBy!=0]
    dBz=dBz[dBz!=0]
    dB = dB[dB != 0]
    return dBx, dBy, dBz, dB
def drawdB(Bpred,Breal,path):
    dBx, dBy, dBz, dB = getdB(Bpred, Breal)
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
    plt.savefig(f'{path}/hist_result.png')
    plt.show()
    plt.close()
def Eval(model, config, field,mode):
    if(mode=='train' or mode=='eval'):
        path  = config['path']
        mode  = config['geo']
        Btype = config['Btype']
        L = config['length']/2*0.25
        N_val =10
        y_test_np_grid = np.linspace(-L, L, N_val)
        if(mode=='cube'):
            x_test_np_grid = np.linspace(-L, L, N_val)
        if(mode=='slice'):
            x_test_np_grid = np.linspace(-0, 0, N_val)
        z_test_np_grid = np.linspace(-L, L, N_val)
        xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, sparse=False)
        x_test_np = xx.reshape((N_val**3, 1))
        y_test_np = yy.reshape((N_val**3, 1))
        z_test_np = zz.reshape((N_val**3, 1))
        x_test = torch.tensor(x_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.float32)
        z_test = torch.tensor(z_test_np, dtype=torch.float32)
        eval_data = torch.cat([x_test, y_test, z_test], axis = 1)
        if(Btype=='Helmholtz'):
            temp_final = np.array([field.HelmholtzB(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
        elif(Btype=='normal'):
            temp_final = np.array([field.B(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
        elif(Btype=='reccirc'):
            temp_final = np.array([field.reccircB(x_test_np,y_test_np,z_test_np)])
        model_output=model.eval(eval_data,'mean')
           
        temp_final = temp_final.reshape(N_val**3, 3)
        drawdB(model_output,temp_final,path)


        model_output = model_output.reshape(N_val, N_val, N_val, 3)
        temp_final = temp_final.reshape(N_val, N_val, N_val, 3)
        ax = ['Bx', 'By', 'Bz']
        idxlist=np.linspace(0,N_val-1,5,dtype=int)
        #在不同的z方向上看xy平面的结果
        for i in range(3):
            figure = plt.figure(figsize=(20,30))
            pred = model_output[:,:,:,i].detach().numpy() 
            real = temp_final[:,:,:,i]
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
                plt.title(f'pred field at z={np.round(-L+idx*L*2/N_val,2)}') 
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
            plt.savefig(f'{path}/slice_result_z_{ax[i]}.png')
            plt.show()
            plt.close()

        #在不同的y方向上看zx平面的结果
        for i in range(3):
            figure = plt.figure(figsize=(20,30))
            pred = model_output[:,:,:,i].detach().numpy()
            real = temp_final[:,:,:,i]
            for j in range(5):
                idx=idxlist[j]
                pred_slice = pred[idx,:,:]                       
                real_slice = real[idx,:,:]
                figure.add_subplot(5,3,j*3+1)
                #plt.rc('font', size=16)
                X = xx[idx,:,:]
                Z = zz[idx,:,:]
                CS = plt.contourf(X,Z, pred_slice, N_val, cmap='jet')
                plt.colorbar(CS)
                plt.xlabel('x')
                plt.ylabel('z')
                plt.title(f'pred field at y={np.round(-L+idx*L*2/N_val,2)}')
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
            plt.savefig(f'{path}/slice_result_y_{ax[i]}.png')
            plt.show()
            plt.close()            

        #在不同的x方向上看yz平面的结果
        for i in range(3):
            figure = plt.figure(figsize=(20,30))
            pred = model_output[:,:,:,i].detach().numpy()
            real = temp_final[:,:,:,i]
            for j in range(5):
                idx=idxlist[j]
                pred_slice = pred[:,idx,:]
                real_slice = real[:,idx,:]
                figure.add_subplot(5,3,j*3+1)
                #plt.rc('font', size=16)
                Y = yy[:,idx,:]
                Z = zz[:,idx,:]
                CS = plt.contourf(Y,Z, pred_slice, N_val, cmap='jet')
                plt.colorbar(CS)
                plt.xlabel('y')
                plt.ylabel('z')
                plt.title(f'pred field at x={np.round(-L+idx*L*2/N_val,decimals=2)}')
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
            plt.savefig(f'{path}/slice_result_x_{ax[i]}.png')                                        
            plt.show()
            plt.close()
    if(mode=='import'):
        path  = config['path']
        data=field[0]
        temp_final=field[1].detach().numpy()
        model_output=model.eval(data,'adjust_nearest')
        if np.any(temp_final == 0):
            print("Warning: temp_final contains zero values.")
        drawdB(model_output,temp_final,path)
        model_output_error=model.eval([data,torch.ones_like(data)*0.001],'error_MonteCarlo')
        print(model_output_error)



        