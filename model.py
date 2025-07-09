import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import copy
#定义梯度计算函数，torch的grad函数返回的数组形状与x相同，结果为du_i/dx_j*grad_outputs_i对i求和。
def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]

class PINN(nn.Module):
    #PINN模型类，内含三个模型B，phi和hard，分别是矢量磁场模型，标量磁势模型和标势硬约束模型。其中标势硬约束模型代码已完成，但基于计算困难未完整实现

    def __init__(self, units,model_mode,train_data,train_labels,layers,activation=torch.sin,activation_grad=torch.cos):
        #模型将data和labels保存在内部
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
        #根据模型模式设计第一层和最后一层输出的维度
        last=3 if model_mode=='B' else 1
        first=3
        
        super(PINN, self).__init__()
        a=[nn.Linear(first, units)]
        for _ in range(layers):
            a.append(nn.Linear(units, units))
        a.append(nn.Linear(units, last))
        #利用nn.ModuleList将所有层存储在一个列表中，方便后续调用
        self.layerlist=nn.ModuleList(a)
        #将激活函数和模型模式保存为类属性，方便后续调用，可以手动调整激活函数，但必须同步调整其梯度函数
        self.activation = activation
        self.model_mode=model_mode
        self.activation_grad=activation_grad
        self.units=units

    def forward(self, inputs,error_prediction=False):
        #前向传播函数，输入为inputs，输出为模型预测的磁场或磁势，若error prediction为True，则输入为[inputs, error]，输出每个坐标点的误差
        #erro_prediction不会被外部触及，其利用误差传播公式计算坐标误差引起的磁场误差，在model.eval('error')时被调用
        #整理数据，inputs[:,0]是一维数组，必须要用view扩展到二维。将三个坐标分离开方便计算梯度
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
        #hlist1和hlist2分别存储线性层的输出和激活函数的输出，每一层的输入都是前面所有激活函数层输出之和。跨层链接可以避免梯度消失问题。
        hlist1=[]
        hlist2=[]
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
        if(self.model_mode=='B' and error_prediction==False):
            return output
        #硬约束模型为了确保边界损失为0，需要额外对输出进行复杂的处理，这个处理是导致逆向传播无法进行的根源。
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
        #两个标势模型在这里都需要求梯度才能得到磁场，这导致反向传播需要求二阶导，速度相比矢量模型异常缓慢
        if((self.model_mode=='phi' or self.model_mode=='hard') and error_prediction==False):
            if(self.model_mode=='hard'):
                output=output*torch.exp(logu)+v
            B_x = gradients(output, input_x)
            B_y = gradients(output, input_y)
            B_z = gradients(output, input_z)
            
            return torch.cat((B_x,B_y,B_z),axis=1)
        #如果是误差预测模式，则通过方均根误差传播公式和激活函数梯度计算误差传播。
        #若z=a*x+b*y+c,则z_error=sqrt(a^2*x_error^2+b^2*y_error^2)
        #由于模型有跨层链接，误差也需要按照同样的方式跨层传播，故初始化errorlist记录每一层误差累加计算下一层的误差
        elif((self.model_mode=='B' or self.model_mode=='phi') and error_prediction==True):
            error=inputs[1]
            errorlist=[]
            for i in range(len(self.layerlist)):      
                #复制当前层，避免修改原始层的参数,并将当前层的bias置为0，weight平方化。这样就可以不用另外写前向传播计算误差。          
                temp_layer=copy.deepcopy(self.layerlist[i])
                temp_layer.bias.data.zero_()  
                temp_layer.weight.data=temp_layer.weight.data**2             
                if(i==len(self.layerlist)-1):
                    erroroutput=torch.sqrt(temp_layer(error**2))
                    break
                errorlist.append(torch.sqrt(temp_layer(error**2)))

                errorlist[i] = torch.abs(self.activation_grad(hlist1[i])*errorlist[i])
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
    #这是计算PINN损失的类，继承自nn.Module。它计算磁场的散度、旋度、边界条件下的散度和旋度，以及预测磁场和真实磁场之间的误差。
    def __init__(self, N_f, L, device, addBC,Lambda):
        super(PINN_Loss, self).__init__() 
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
        #仍旧需要将磁场的每一列拆分开，并确保其为二维数组
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
        #计算磁场的散度：div B = ∇·B = dBx/dx + dBy/dy + dBz/dz
        loss_BC_div = torch.mean(torch.square(dx+dy+dz))
        #随机抽点
        y_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        #在某些情形下，抽的点只在平面上
        if(train_y.max()>0):
            x_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        else:
            x_f = np.random.default_rng().uniform(low = -self.L/10, high = self.L/10, size = ((self.N_f, 1)))
        z_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        #张量化后才能开启梯度计算
        self.x_f = torch.tensor(x_f, dtype = torch.float32,device=device,requires_grad=True)    
        self.y_f = torch.tensor(y_f, dtype = torch.float32,device=device,requires_grad=True)
        self.z_f = torch.tensor(z_f, dtype = torch.float32,device=device,requires_grad=True)
        temp_pred = model(torch.cat((self.x_f, self.y_f, self.z_f), axis=1))
        temp_ux = temp_pred[:,0].requires_grad_(True)
        temp_uy = temp_pred[:,1].requires_grad_(True)
        temp_uz = temp_pred[:,2].requires_grad_(True)
        #散度
        u_x = gradients(temp_ux, self.x_f)
        u_y = gradients(temp_uy, self.y_f)
        u_z = gradients(temp_uz, self.z_f)
        #计算旋度损失作为loss_cross
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
            #对于标势模型，旋度和边界条件下的旋度都不需要计算
            loss_cross = torch.tensor(0)
            loss_BC_cul = torch.tensor(0)
        #计算散度的平方作为loss_f
        loss_f = torch.mean(torch.square(u_x + u_y + u_z))
        
        #计算采样磁场大小和预测磁场大小的差，作为loss_u
        loss_u = torch.mean(torch.square(pred - labels))
        if(self.addBC):
            loss = loss_f*self.Lambda + loss_u + loss_cross*self.Lambda + loss_BC_div*self.Lambda + loss_BC_cul*self.Lambda
        else:
            loss  = loss_f*self.Lambda + loss_u + loss_cross*self.Lambda
        return loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss
#MODELS类用于管理多个PINN模型的集合，提供评估和保存功能。
#它可以评估给定数据的输出，支持多种评估模式，如平均输出、最近邻输出、调整后的最近邻输出、误差预测等。
class MODELS():
    def __init__(self, config,train_data,train_labels):
        self.N_models = config['N_models']
        self.models = []
        self.config = config
        #将mean和std写入内部，在eval时方便进行标准化和去标准化，确保外部一行代码调用。
        self.config['mean_data'] = torch.tensor(self.config['mean_data'],dtype=torch.float32)
        self.config['std_data'] = torch.tensor(self.config['std_data'],dtype=torch.float32)
        self.config['mean'] = torch.tensor(self.config['mean'],dtype=torch.float32)
        self.config['std'] = torch.tensor(self.config['std'],dtype=torch.float32)
        self.train_data = train_data
        self.train_labels = train_labels
    def eval(self, eval_data,eval_mode='mean'):
        #eval_data是坐标，必要时也可以是[坐标，坐标/磁场误差]
        #eval_mode可以是'mean'，'nearest'，'adjust_nearest'，'error_MonteCarlo'，'error_field'或'error'
        #如果eval_mode为'mean'，则返回所有模型的平均输出
        #如果eval_mode为'nearest'，则返回在离eval_data最近的训练坐标上表现最好模型的输出
        #如果eval_mode为'adjust_nearest'，则返回根据离eval_data最近的训练坐标上的表现加权平均的各模型输出，权重为各模型在该最近邻坐标上预测的磁场与真实磁场的偏差的倒数。
        #如果eval_mode为'error_MonteCarlo'，则返回在eval_data的误差范围内随机采样100次的方均根输出
        #如果eval_mode为'error_field'，则使用扰动后的磁场微调5个模型，并使用这5个模型在eval_data上的偏差的方均根作为输出。
        #如果eval_mode为'error'，则根据误差传递公式计算eval_data的误差传播。
        #error和error_MonteCarlo计算的是来源相同的误差，即坐标测量误差引起的预测磁场误差。
        #区别在于误差传播公式和montecarlo方法，且error使用的是坐标测量的标准差（不假设分布），而error_MonteCarlo使用的是坐标测量的允差（最大误差）并假设均匀分布。
        #同样，error_field使用的是磁场测量的允差并假设均匀分布。并且同样为montecarlo方法。
        for i in range(self.N_models):
            self.models[i].eval()
            self.models[i].to('cpu')
        #对数据和误差标准化
        if(eval_mode=='mean' or eval_mode=='nearest' or eval_mode=='adjust_nearest'):
            eval_data = (eval_data-self.config['mean_data'])/self.config['std_data']
        elif(eval_mode=='error'):
            eval_data[0] = (eval_data[0]-self.config['mean_data'])/self.config['std_data'] 
            eval_data[1] = eval_data[1]/self.config['std_data'] 
        if(eval_mode=='mean'):
            model_output = torch.zeros((eval_data.shape[0], 3))
            for i in range(self.N_models):
                self.models[i].eval()
                model_output = model_output + self.models[i](eval_data)*self.config['std']+self.config['mean']
            model_output = model_output/self.N_models
        if(eval_mode=='nearest'):
            #此处可以不使用循环而是利用额外维度矢量化操作，获取更快速度
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
                #由于微调模型本身就会带来一定波动，故做一个0磁场误差的对照组，并在最终结果中减去对照组的波动
                print('start evaluate baseline')
                baseline=self.eval([eval_data_base,torch.zeros(field_data_error.shape)],eval_mode='error_field')
            for i in range(5*self.N_models):
                random_data=torch.rand(field_data_error.shape)
                field_data=(self.train_labels+field_data_error*random_data-self.config['mean'])/self.config['std']
                model1=copy.deepcopy(self.models[i//5])
                #使用低学习率，确保微调后的模型不会偏离原有模型太远
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
                    print(f'evaluating field loss:{j,loss_u}')
                model1=copy.deepcopy(bestmodel)
                model1.eval()
                model_output=model1((eval_data_base-self.config['mean_data'])/self.config['std_data'])*self.config['std']+self.config['mean']   
                output_error_new=torch.abs(output_base-model_output)
                output_error=output_error+output_error_new**2
            output_error=(output_error/5)**0.5
            if(~(torch.sum(field_data_error**2)==0)):
                #减去对照组的波动，把小于0的位置置为0
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