import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import copy
def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]
class PINN(nn.Module):
    def __init__(self, units,model_mode,train_data,train_labels,layers,activation=torch.sin,activation_grad=torch.cos):
        self.train_data=train_data
        self.train_labels=train_labels
        if(model_mode=='hard'):
            a=self.train_data.reshape(1,-1,3).repeat(self.train_data.shape[0],1,1)
            b=self.train_data.reshape(-1,1,3)
            c=torch.sum((a-b)**2,axis=2)
            c[c==0]=1
            logc=torch.sum(torch.log(c),axis=1)
            self.logc=logc
            self.logu0=torch.sum(torch.log(torch.sum(self.train_data**2,axis=1)),dim=0)
        last=3 if model_mode=='B' else 1
        first=3
        
        super(PINN, self).__init__()
        a=[nn.Linear(first, units)]
        for i in range(layers):
            a.append(nn.Linear(units, units))
        a.append(nn.Linear(units, last))
        self.layerlist=nn.ModuleList(a)
        #self.hidden_layer6 = nn.Linear(units, units)
        #self.hidden_layer7 = nn.Linear(units, units)        
        #self.hidden_layer8 = nn.Linear(units, 3)
        self.activation = activation
        self.model_mode=model_mode
        self.activation_grad=activation_grad
        self.units=units

    def forward(self, inputs,error_prediction=False):
               
        if(self.model_mode=='B' and error_prediction==False):
            input=inputs
        if((self.model_mode=='phi' or self.model_mode=='hard') and error_prediction==False):
            if(len(inputs.shape)==1):
                inputs=inputs.view(1,-1)
            input_x = inputs[:,0].view(-1,1).requires_grad_(True)
            input_y = inputs[:,1].view(-1,1).requires_grad_(True)
            input_z = inputs[:,2].view(-1,1).requires_grad_(True)
            input=torch.cat((input_x,input_y,input_z),axis=1)
        if(self.model_mode=='B' and error_prediction==True):
            input=inputs[0]
        if(self.model_mode=='phi' and error_prediction==True):
            input_x=inputs[0][:,0].view(-1,1).requires_grad_(True)
            input_y=inputs[0][:,1].view(-1,1).requires_grad_(True)
            input_z=inputs[0][:,2].view(-1,1).requires_grad_(True)
            input=torch.cat((input_x,input_y,input_z),axis=1)
        hlist1=[]
        hlist2=[]
        hlist11,hlist22=[],[]
        
        for i in range(len(self.layerlist)):            
            if(i==len(self.layerlist)-1):
                output=self.layerlist[i](input) 
                break
            hlist1.append(self.layerlist[i](input))
            hlist2.append(self.activation(hlist1[i]))
            b=0
            for j in range(i+1):
                b=b+hlist2[j] 
            input=b
        if(error_prediction==True):
            input1=inputs[0]+inputs[1]
            for i in range(len(self.layerlist)):            
                if(i==len(self.layerlist)-1):
                    output1=self.layerlist[i](input1) 
                    break
                hlist11.append(self.layerlist[i](input1))
                hlist22.append(self.activation(hlist11[i]))
                b=0
                for j in range(i+1):
                    b=b+hlist22[j] 
                input1=b        
        #h5 = self.activation(h5)
        #h6 = self.hidden_layer6(h5)
        #h6 = self.activation(h6)
        #h7 = self.hidden_layer7(h6)
        #h7 = self.activation(h7)
        #h8 = self.hidden_layer8(h7)

        if(self.model_mode=='B' and error_prediction==False):
            return output
        if(self.model_mode=='hard' and error_prediction==False):
            print((torch.sum((torch.cat((input_x,input_y,input_z),axis=1).view(-1,1,3)-self.train_data.view(1,-1,3))**2,dim=2)==0).any())
            logu=torch.sum(torch.log(torch.sum((torch.cat((input_x,input_y,input_z),axis=1).view(-1,1,3)-self.train_data.view(1,-1,3))**2,dim=2)),dim=1)
            logu=logu-self.logu0
            x=torch.cat((input_x,input_y,input_z),axis=1).view(1,-1,1,3).repeat(self.train_data.shape[0],1,1,1)
            y=self.train_data.view(1,1,-1,3).repeat(self.train_data.shape[0],1,1,1)
            z=torch.sum((x-y)**2,axis=3)
            z[torch.linspace(0,self.train_data.shape[0]-1,self.train_data.shape[0],dtype=int),:,torch.linspace(0,self.train_data.shape[0]-1,self.train_data.shape[0],dtype=int)]=1
            logz=torch.sum(torch.log(z),axis=2)
            a=torch.sum((x-self.train_data.view(self.train_data.shape[0],1,1,3))*self.train_labels.view(self.train_data.shape[0],1,1,3),dim=(2,3))
            v=torch.sum(a*torch.exp(logz-self.logc),axis=0)
        if((self.model_mode=='phi' or self.model_mode=='hard') and error_prediction==False):
            if(self.model_mode=='hard'):
                output=output*torch.exp(logu)+v
            B_x = gradients(output, input_x)
            B_y = gradients(output, input_y)
            B_z = gradients(output, input_z)
            
            return torch.cat((B_x,B_y,B_z),axis=1)

        elif((self.model_mode=='B' or self.model_mode=='phi') and error_prediction==True):
            error=inputs[1]
            errorlist=[]
            for i in range(len(self.layerlist)):                

                temp_layer=copy.deepcopy(self.layerlist[i])
                temp_layer.bias.data.zero_()  
                temp_layer.weight.data=temp_layer.weight.data**2             
                if(i==len(self.layerlist)-1):
                    erroroutput=torch.sqrt(temp_layer(error**2))
                    print(erroroutput,output1-output)
                    break
                errorlist.append(torch.sqrt(temp_layer(error**2)))

                errorlist[i] = torch.abs(self.activation_grad(hlist1[i])*errorlist[i])
                #print(errorlist[i],hlist22[i]-hlist2[i],i)
                f=0
                for j in range(i+1):
                    f=f+errorlist[j]
                error=f
            if(self.model_mode=='B'):
                return erroroutput
            if(self.model_mode=='phi'):
                B_x_error=gradients(erroroutput, input_x)
                B_y_error=gradients(erroroutput, input_y)
                B_z_error=gradients(erroroutput, input_z)
                return torch.cat((B_x_error,B_y_error,B_z_error),axis=1)

class PINN_Loss(nn.Module):
    #初始化神经网络输入，定义输入参数
    def __init__(self, N_f, L, device, addBC,Lambda):
        super(PINN_Loss, self).__init__() #继承tf.keras.Model的功能
        self.N_f = N_f
        self.L = L
        self.device = device
        if(addBC==0):
            self.addBC = False
        if(addBC==1):
            self.addBC = True
        self.Lambda=Lambda

    def forward(self, data, pred, labels, model):
        device = self.device
        train_x = data[:,0].view(-1,1).requires_grad_(True)
        train_y = data[:,1].view(-1,1).requires_grad_(True)
        train_z = data[:,2].view(-1,1).requires_grad_(True)
        B = model(torch.cat((train_x, train_y, train_z), axis=1))
        B_x = B[:,0].requires_grad_(True)
        B_y = B[:,1].requires_grad_(True)
        B_z = B[:,2].requires_grad_(True)
        dx = gradients(B_x, train_x)
        dy = gradients(B_y, train_y)
        dz = gradients(B_z, train_z)
                
        loss_BC_div = torch.mean(torch.square(dx+dy+dz))
        

        y_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        if(train_y.max()>0):
            x_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        else:
            x_f = np.random.default_rng().uniform(low = -self.L/10, high = self.L/10, size = ((self.N_f, 1)))
        z_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        self.x_f = torch.tensor(x_f, dtype = torch.float32,device=device,requires_grad=True)    
        self.y_f = torch.tensor(y_f, dtype = torch.float32,device=device,requires_grad=True)
        self.z_f = torch.tensor(z_f, dtype = torch.float32,device=device,requires_grad=True)
        temp_pred = model(torch.cat((self.x_f, self.y_f, self.z_f), axis=1))
        temp_ux = temp_pred[:,0].requires_grad_(True)
        temp_uy = temp_pred[:,1].requires_grad_(True)
        temp_uz = temp_pred[:,2].requires_grad_(True)
        u_x = gradients(temp_ux, self.x_f)
        u_y = gradients(temp_uy, self.y_f)
        u_z = gradients(temp_uz, self.z_f)
        if(model.model_mode=='B'):
            dzy = gradients(B_z, train_y)
            dzx = gradients(B_z, train_x)
            dyz = gradients(B_y, train_z)
            dyx = gradients(B_y, train_x)
            dxy = gradients(B_x, train_y)
            dxz = gradients(B_x, train_z)
            loss_BC_cul = torch.mean(torch.square(dzy - dyz) + torch.square(dxz - dzx) + torch.square(dyx - dxy))
            u_zy = gradients(temp_uz, self.y_f) #dBz_f/dy_f
            u_zx = gradients(temp_uz, self.x_f) #dBz_f/dx_f
            u_yz = gradients(temp_uy, self.z_f) #dBy_f/dz_f
            u_yx = gradients(temp_uy, self.x_f) #dBy_f/dx_f
            u_xz = gradients(temp_ux, self.z_f) #dBx_f/dz_f
            u_xy = gradients(temp_ux, self.y_f) #dBx_f/dy_f
            loss_cross = torch.mean(torch.square(u_zy - u_yz) + torch.square(u_xz - u_zx) + torch.square(u_yx - u_xy))
        else:
            loss_cross = torch.tensor(0)
            loss_BC_cul = torch.tensor(0)
        #计算散度：div B = ∇·B = dBx_f/dx_f + dBy_f/dy_f + dBz_f/dz_f
        #计算散度的平方作为loss_∇·B 
        loss_f = torch.mean(torch.square(u_x + u_y + u_z))
        #计算旋度的模方：|∇×B|^2，作为loss_∇×B
        
        #计算采样磁场大小和预测磁场大小的差，作为loss_B
        loss_u = torch.mean(torch.square(pred - labels))
        if(self.addBC):
            loss = loss_f*self.Lambda + loss_u + loss_cross*self.Lambda + loss_BC_div*self.Lambda + loss_BC_cul*self.Lambda
        else:
            loss  = loss_f*self.Lambda + loss_u + loss_cross*self.Lambda
        return loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss
class MODELS():
    def __init__(self, config,train_data,train_labels):
        self.N_models = config['N_models']
        self.models = []
        self.config = config
        self.config['mean_data'] = torch.tensor(self.config['mean_data'],dtype=torch.float32)
        self.config['std_data'] = torch.tensor(self.config['std_data'],dtype=torch.float32)
        self.config['mean'] = torch.tensor(self.config['mean'],dtype=torch.float32)
        self.config['std'] = torch.tensor(self.config['std'],dtype=torch.float32)
        self.train_data = train_data
        self.train_labels = train_labels
    def eval(self, eval_data,eval_mode='mean'):
        for i in range(self.N_models):
            self.models[i].eval()
        if(eval_mode=='mean' or eval_mode=='nearest' or eval_mode=='adjust_nearest'):
            eval_data = (eval_data-self.config['mean_data'])/self.config['std_data']
        elif(eval_mode=='error'):
            eval_data[0] = (eval_data[0]-self.config['mean_data'])/self.config['std_data'] 
            eval_data[1] = eval_data[1]/self.config['std_data'] 
        for i in range(self.N_models):
            self.models[i].to('cpu')
        if(eval_mode=='mean'):
            model_output = torch.zeros((eval_data.shape[0], 3))
            for i in range(self.N_models):
                self.models[i].eval()
                model_output = model_output + self.models[i](eval_data)*self.config['std']+self.config['mean']
            model_output = model_output/self.N_models
        if(eval_mode=='nearest'):
            model_output = torch.zeros((eval_data.shape[0], 3))
            for j in range(eval_data.shape[0]):
                maxidx=torch.argmin(torch.sum(torch.square(self.train_data-eval_data[j,:]),axis=1))
                nearest_data=self.train_data[maxidx,:]
                nearest_labels=self.train_labels[maxidx,:]
                min_delta=100000000
                for i in range(self.N_models):
                    delta=torch.linalg.vector_norm((self.models[i](nearest_data)-nearest_labels))
                    if(delta<min_delta):
                        min_delta=delta
                        min_idx=i
                model_output[j,:]=self.models[min_idx](eval_data[j,:])*self.config['std']+self.config['mean']
        if(eval_mode=='adjust_nearest'):
            model_output = torch.zeros((eval_data.shape[0], 3))
            for j in range(eval_data.shape[0]):
                maxidx=torch.argmin(torch.sum(torch.square(self.train_data-eval_data[j,:]),axis=1))
                nearest_data=self.train_data[maxidx,:]
                nearest_labels=self.train_labels[maxidx,:]
                delta=[]
                for i in range(self.N_models):
                    delta.append(torch.linalg.vector_norm((self.models[i](nearest_data)-nearest_labels)))
                delta=torch.tensor(delta)
                delta=delta**(-1)
                delta=delta/torch.sum(delta)
                for i in range(self.N_models):
                    model_output[j,:]=model_output[j,:]+(self.models[i](eval_data[j,:])*self.config['std']+self.config['mean'])*delta[i]
        if(eval_mode=='error_MonteCarlo'):
            eval_data_base=eval_data[0]
            eval_data_error=eval_data[1]
            output_base=self.eval(eval_data_base,eval_mode='mean') 
            output_error=torch.zeros(output_base.shape)
            for i in range(100):
                random_data=torch.rand(eval_data_error.shape)
                eval_data=eval_data_base+eval_data_error*random_data
                model_output=self.eval(eval_data,eval_mode='mean')
                output_error_new=torch.abs(output_base-model_output)
                output_error=output_error+output_error_new**2
            model_output=(output_error/100)**0.5
        if(eval_mode=='error'):
            model_output = torch.zeros((eval_data[0].shape[0], 3))
            for i in range(self.N_models):
                self.models[i].eval()
                model_output = model_output + self.models[i](eval_data,True)*self.config['std']
            model_output = model_output/self.N_models
        if(eval_mode=='error_field'):
            eval_data_base=eval_data[0]
            field_data_error=eval_data[1]
            output_base=self.eval(eval_data_base,eval_mode='mean')
            output_error=torch.zeros(output_base.shape)
            if(~(torch.sum(field_data_error**2)==0)):
                print('start evaluate baseline')
                baseline=self.eval([eval_data_base,torch.zeros(field_data_error.shape)],eval_mode='error_field')
            for i in range(5*self.N_models):
                random_data=torch.rand(field_data_error.shape)
                field_data=(self.train_labels+field_data_error*random_data-self.config['mean'])/self.config['std']
                model1=copy.deepcopy(self.models[i//5])
                optimizer = torch.optim.Adam(model1.parameters(), lr=0.0003) 
                model1.train()
                criterion = PINN_Loss(self.config['Npde'], self.config['length'], 'cpu', self.config['addBC'], self.config['Lambda'])
                bestloss=1000000000
                for j in range(200):
                    optimizer.zero_grad()
                    pred = model1((self.train_data-self.config['mean_data'])/self.config['std_data'])
                    loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion((self.train_data-self.config['mean_data'])/self.config['std_data'], pred, field_data, model1)
                    loss.backward()
                    optimizer.step()
                    if(loss<bestloss):
                        bestloss=loss
                        bestmodel=copy.deepcopy(model1)
                    print(j,loss_u)
                model1=copy.deepcopy(bestmodel)
                model1.eval()
                model_output=model1((eval_data_base-self.config['mean_data'])/self.config['std_data'])*self.config['std']+self.config['mean']   
                output_error_new=torch.abs(output_base-model_output)
                output_error=output_error+output_error_new**2
                print(i)
            output_error=(output_error/5)**0.5
            print(output_error)
            if(~(torch.sum(field_data_error**2)==0)):
                output_error=output_error**2-baseline**2
                output_error[output_error<0]=0
                output_error=output_error**0.5
            model_output=output_error
        return model_output

    def save(self, path):
        for i in range(self.N_models):
            torch.save(self.models[i].state_dict(), path + f'/best_model{i}.pt')
    def load(self, path):
        for i in range(self.N_models):
            self.models.append(PINN(self.config['units'],self.config['model_mode'],self.train_data,self.train_labels,self.config['layers']))
            self.models[i].load_state_dict(torch.load(path + f'/best_model{i}.pt'))