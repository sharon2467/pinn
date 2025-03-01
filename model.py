import torch
from torch import nn
from torch.autograd import grad
from torch import optim
import numpy as np

def sine_activation(x):
    return torch.tanh(x)
def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]
class PINN(nn.Module):
    def __init__(self, units,model_mode,train_data,train_labels,activation=sine_activation):
        if(model_mode=='hard'):
            self.train_data=train_data
            self.train_labels=train_labels
        last=3 if model_mode=='B' else 1
        first=2 if model_mode=='coil' else 3
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(first, units)
        self.hidden_layer2 = nn.Linear(units, units)
        self.hidden_layer3 = nn.Linear(units, units)
        self.hidden_layer4 = nn.Linear(units, units)
        self.hidden_layer5 = nn.Linear(units, last)
        #self.hidden_layer6 = nn.Linear(units, units)
        #self.hidden_layer7 = nn.Linear(units, units)        
        #self.hidden_layer8 = nn.Linear(units, 3)
        self.activation = activation
        self.model_mode=model_mode

    def forward(self, inputs):
               
        #h5 = self.activation(h5)
        #h6 = self.hidden_layer6(h5)
        #h6 = self.activation(h6)
        #h7 = self.hidden_layer7(h6)
        #h7 = self.activation(h7)
        #h8 = self.hidden_layer8(h7)
        if(self.model_mode=='B' ):
            h1 = self.hidden_layer1(inputs)
            h1 = self.activation(h1)
            h2 = self.hidden_layer2(h1)
            h2 = self.activation(h2)
            h3 = self.hidden_layer3(h2+h1)
            h3 = self.activation(h3)
            h4 = self.hidden_layer4(h3+h2+h1)
            h4 = self.activation(h4)
            h5 = self.hidden_layer5(h4+h3+h2+h1) 
            return h5
        elif(self.model_mode=='phi' or 'hard'):
            if(len(inputs.shape)==1):
                inputs=inputs.view(1,-1)
            input_x = inputs[:,0].view(-1,1).requires_grad_(True)
            input_y = inputs[:,1].view(-1,1).requires_grad_(True)
            input_z = inputs[:,2].view(-1,1).requires_grad_(True)
            h1=self.hidden_layer1(torch.cat((input_x, input_y, input_z), axis=1))
            h1=self.activation(h1)  
            h2=self.hidden_layer2(h1)
            h2=self.activation(h2)
            h3=self.hidden_layer3(h2+h1)
            h3=self.activation(h3)
            h4=self.hidden_layer4(h3+h2+h1)
            h4=self.activation(h4)
            h5=self.hidden_layer5(h4+h3+h2+h1)
            if(self.model_mode=='hard'):
                u=torch.prod(torch.sum((torch.cat((input_x,input_y,input_z),axis=1)-self.train_data.view(1,3,-1))**2,dim=2),dim=3)
                v=None
            B_x = gradients(h5, input_x)
            B_y = gradients(h5, input_y)
            B_z = gradients(h5, input_z)
            return torch.cat((B_x,B_y,B_z),axis=1)

# class PINN1(nn.Module):
#     def __init__(self,units):
#         super(PINN1, self).__init__()
#         self.pinn1=PINN(units,'Bz')
#         self.pinn2=PINN(units,'coil')
#         self.pinn3=PINN(units,'coil')
#         self.model_mode='coil'
#     def forward(self,inputs):
#         if(len(inputs.shape)==1):
#             inputs=inputs.view(1,-1)
#         train_x = inputs[:,0].view(-1,1).requires_grad_(True)
#         train_y = inputs[:,1].view(-1,1).requires_grad_(True)
#         train_z = inputs[:,2].view(-1,1).requires_grad_(True)
#         rxz=train_x**2+train_z**2
#         ryz=train_y**2+train_z**2
#         phi=self.pinn1(torch.cat((train_x,train_y,train_z),axis=1))+self.pinn2(torch.cat((rxz,train_y),axis=1))+self.pinn3(torch.cat((ryz,train_x),axis=1))
#         B_x = gradients(phi, train_x)
#         B_y = gradients(phi, train_y)
#         B_z = gradients(phi, train_z)
#         return torch.cat((B_x,B_y,B_z),axis=1)

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
        self.x_f = torch.tensor(x_f, dtype = torch.float32).to(device).requires_grad_(True)
        self.y_f = torch.tensor(y_f, dtype = torch.float32).to(device).requires_grad_(True)
        self.z_f = torch.tensor(z_f, dtype = torch.float32).to(device).requires_grad_(True)
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
        eval_data = (eval_data-self.config['mean_data'])/self.config['std_data']
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
        return model_output

    def save(self, path):
        for i in range(self.N_models):
            torch.save(self.models[i].state_dict(), path + f'/best_model{i}.pt')
    def load(self, path):
        for i in range(self.N_models):
            self.models.append(PINN(self.config['units']))
            self.models[i].load_state_dict(torch.load(path + f'/best_model{i}.pt'))