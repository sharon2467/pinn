import numpy as np
import torch
import os
import scipy.special as sc

class data_generation:
    def __init__(self, radius=1, N_sample=3, N_test=1000, L=0.5,dx=2,dy=2,dz=2,a=2,b=2,Ix=0.5,Iy=0.5,Iz=10,radius1=1,radius2=1):
        self.radius = radius
        self.N_sample = N_sample
        self.N_test = N_test
        self.L = L
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.a=a
        self.b=b
        self.Ix=Ix
        self.Iy=Iy
        self.Iz=Iz
        self.radius1=radius1
        self.radius2=radius2

    def circB(self,x,y,z):
         # 定义一些参数，用于公式中
        x_prime=x
        y_prime=y
        z_prime=z
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = self.radius**2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime)*self.radius
        beta_sq_prime = self.radius**2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime)*self.radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        # 计算旋转后的磁场分量
        Bx_prime = x_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 1 / (2*alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)
    
        #return np.concatenate([Bx.reshape(-1,1),By.reshape(-1,1),Bz.reshape(-1,1)], axis=1)*-100
        return np.array([Bx_prime,By_prime,Bz_prime])*-100
    def lineB(self,pos,start,end):
        start=np.expand_dims(start,0)
        end=np.expand_dims(end,0)
        r12=end-start
        r10=pos-start
        r20=pos-end
        cos1=np.dot(r10,np.transpose(r12))/np.expand_dims(np.linalg.norm(r12)*np.linalg.norm(r10,axis=1),1)
        cos2=np.dot(r20,np.transpose(r12))/np.expand_dims(np.linalg.norm(r12)*np.linalg.norm(r20,axis=1),1)
        r=np.linalg.norm(np.cross(r10,r20),axis=1)/np.linalg.norm(r12)
        
        B=np.cross(r10,r20)
        #print(np.expand_dims(np.linalg.norm(B,axis=1),1).shape,np.expand_dims(r,1).shape,cos1.shape,cos2.shape)
        
        B=B/np.expand_dims(np.linalg.norm(B,axis=1),1)/np.expand_dims(r,1)*(cos1-cos2)/4/np.pi
        return B
    def recB_xy(self,x,y,z):
        #x轴长为a，y轴长为b
        pos1=np.array([self.a/2,self.b/2,0])
        pos2=np.array([-self.a/2,self.b/2,0])
        pos3=np.array([-self.a/2,-self.b/2,0])
        pos4=np.array([self.a/2,-self.b/2,0])
        pos=np.concatenate((x,y,z),axis=1)
        B=self.lineB(pos,pos1,pos2)+self.lineB(pos,pos2,pos3)+self.lineB(pos,pos3,pos4)+self.lineB(pos,pos4,pos1)
        return B
    def circB_xy(self,x,y,z):
        #Defining some parameters to be used in the formulas
         # 定义一些参数，用于公式中
        x_prime=x
        y_prime=y
        z_prime=z
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = self.radius**2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime)*self.radius
        beta_sq_prime = self.radius**2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime)*self.radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        # 计算旋转后的磁场分量
        Bx_prime = x_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 1 / (2*alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)

        #return np.concatenate([Bx.reshape(-1,1),By.reshape(-1,1),Bz.reshape(-1,1)], axis=1)*-100
        return np.array([Bx_prime,By_prime,Bz_prime])

    def circB_yz(self, x, y, z):
    
        # 坐标转换，将线圈旋转到yz平面
        x_prime = z  # x' = z
        y_prime = y  # y' = y
        z_prime = -x # z' = -x
    
        # 定义一些参数，用于公式中
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = self.radius**2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime)*self.radius
        beta_sq_prime = self.radius**2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime)*self.radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        # 计算旋转后的磁场分量
        Bx_prime = x_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 1 / (2*alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)
        # 由于线圈旋转，磁场方向也需要相应地调整
        # 原始函数中Bx, By, Bz对应于xy平面内的磁场分量，现在我们需要将它们转换为yz平面内的分量
        # 旋转90度后，Bx变为Bz，By保持不变，Bz变为-Bx
        Bx_rotated = Bz_prime
        By_rotated = By_prime
        Bz_rotated = -Bx_prime
    
        # 返回旋转后的磁场分量
        return np.array([Bx_rotated, By_rotated, Bz_rotated]) 

    def circB_zx(self, x, y, z):

        # 坐标转换，将线圈旋转到yz平面
        x_prime = x  # x' = x
        y_prime = -z  # y' = -z
        z_prime = y # z' = y

        # 定义一些参数，用于公式中
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = self.radius**2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime)*self.radius
        beta_sq_prime = self.radius**2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime)*self.radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        # 计算旋转后的磁场分量
        Bx_prime = x_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 1 / (2*alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius**2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)
        # 由于线圈旋转，磁场方向也需要相应地调整
        # 原始函数中Bx, By, Bz对应于xy平面内的磁场分量，现在我们需要将它们转换为yz平面内的分量
        # 旋转90度后，Bx变为Bz，By保持不变，Bz变为-Bx
        Bx_rotated = Bx_prime
        By_rotated = -Bz_prime
        Bz_rotated = By_prime

        # 返回旋转后的磁场分量
        return np.array([Bx_rotated, By_rotated, Bz_rotated]) 
    def circB_rotate(self,x,y,z,theta,phi,radius):
        #theta是绕y轴正向旋转的角度，phi是绕z轴正向旋转的角度
        x_prime=x*np.cos(theta)*np.cos(phi)+y*np.cos(theta)*np.sin(phi)-z*np.sin(theta)
        y_prime=-x*np.sin(phi)+y*np.cos(phi)
        z_prime=x*np.sin(theta)*np.cos(phi)+y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = radius**2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime)*radius
        beta_sq_prime = radius**2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime)*radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)
        
        # 计算旋转后的磁场分量
        rho_sq_prime[rho_sq_prime<1e-6]=9999
        Bx_prime = x_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)*(~(rho_sq_prime==9999))
        By_prime = y_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)*(~(rho_sq_prime==9999))

        Bz_prime = 1 / (2*alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((radius**2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)

        Bx=Bx_prime*np.cos(theta)*np.cos(phi)-By_prime*np.sin(phi)+Bz_prime*np.sin(theta)*np.cos(phi)
        By=Bx_prime*np.cos(theta)*np.sin(phi)+By_prime*np.cos(phi)+Bz_prime*np.sin(theta)*np.sin(phi)
        Bz=-Bx_prime*np.sin(theta)+Bz_prime*np.cos(theta)
        return np.concatenate((Bx,By,Bz),axis=1)/np.pi
    def B(self,x,y,z):
        field = 1*self.circB(x + 1.01,y + 1.0,z - 4.0) + 1*self.circB(x - 1.01,y - 1.0, z - 4.0) + 1*self.circB(x + 1.01,y - 1.0,z - 4.0) + 1*self.circB(x - 1.01,y + 1.0,z - 4.0) + 1*self.circB(x + 1.01,y + 1.0,z + 4.0) + 1*self.circB(x - 1.01,y - 1.0,z + 4.0) + 1*self.circB(x + 1.01,y - 1.0,z + 4.0) + 1*self.circB(x - 1.01,y + 1.0,z + 4.0)
        return field.tolist()

    def HelmholtzB(self, x, y, z):
        field  = 0
        field += (self.circB_xy(x, y, z-self.dz/2) + self.circB_xy(x, y, z+self.dz/2))*self.Iz
        field += (self.circB_yz(x-self.dx/2, y, z) + self.circB_yz(x+self.dx/2, y, z))*self.Ix
        field += (self.circB_zx(x, y+self.dy/2, z) + self.circB_zx(x, y-self.dy/2, z))*self.Iy
        return field.tolist()
    def reccircB(self,x,y,z):
        field =0
        field += self.circB_rotate(x+self.dx/2,y,z,np.pi/2,0,self.radius1)*self.Ix
        field += self.circB_rotate(x-self.dx/2,y,z,np.pi/2,0,self.radius1)*self.Ix
        field += self.circB_rotate(x,y+self.dy/2,z,np.pi/2,np.pi/2,self.radius2)*self.Iy
        field += self.circB_rotate(x,y-self.dy/2,z,np.pi/2,np.pi/2,self.radius2)*self.Iy
        field += self.recB_xy(x,y,z-self.dz/2)*self.Iz
        field += self.recB_xy(x,y,z+self.dz/2)*self.Iz
        return field.tolist()
    def number_generator(self,min,max,num,mode):
        if (mode == 'normal'):
            a=np.random.normal(0,1,num)
            a=(a-a.min())/(a.max()-a.min())*(max-min)+min
            a=a.reshape((num,1))
            return a
        if (mode == 'linspace'):
            return np.linspace(min,max,num).reshape((num,1))
        else:
            return np.random.default_rng().uniform(min,max,num).reshape((num,1))
    def train_data_cube(self, Btype='normal', inner_sample=False,mode='uniform'):
        L1 = self.L
        N = self.N_sample
        x1 = np.concatenate((-L1*np.ones([N, 1]), L1*np.ones([N, 1]), #L和-L上各取Nu个点，确保采样点在格子的表面
                        self.number_generator(-L1,L1,4*N,mode)), #在-L到L之间随机生成4*Nu个点
                        axis = 0)
        y1 = np.concatenate((self.number_generator(-L1,L1,2*N,mode), #在-L到L之间随机生成2*Nu个点
                        -L1*np.ones([N, 1]), L1*np.ones([N, 1]),
                        self.number_generator(-L1,L1,2*N,mode)), #在-L到L之间随机生成2*Nu个点
                        axis = 0)
        z1 = np.concatenate((self.number_generator(-L1,L1,4*N,mode), #在-L到L之间随机生成4*Nu个点
                        -L1*np.ones([N, 1]), L1*np.ones([N, 1])),
                        axis = 0)
        x1 = torch.tensor(x1, dtype=torch.float32)
        y1 = torch.tensor(y1, dtype=torch.float32)
        z1 = torch.tensor(z1, dtype=torch.float32)
        pos1 = torch.cat((x1, y1, z1), axis=1)
        if(inner_sample):
            L2 = self.L*0.75
            x2 = np.concatenate((-L2*np.ones([N, 1]), L2*np.ones([N, 1]), 
                            self.number_generator(-L2,L2,4*N,mode)),
                            axis = 0)
            y2 = np.concatenate((self.number_generator(-L2,L2,2*N,mode),
                            -L2*np.ones([N, 1]), L2*np.ones([N, 1]),
                            self.number_generator(-L2,L2,2*N,mode)),
                            axis = 0)
            z2 = np.concatenate((self.number_generator(-L2,L2,4*N,mode),
                            -L2*np.ones([N, 1]), L2*np.ones([N, 1])),
                            axis = 0)
     
            x2 = torch.tensor(x2, dtype=torch.float32)
            y2 = torch.tensor(y2, dtype=torch.float32)
            z2 = torch.tensor(z2, dtype=torch.float32)
            pos2 = torch.cat((x2, y2, z2), axis=1)
            x  = torch.cat((x1,x2))
            y  = torch.cat((y1,y2))
            z  = torch.cat((z1,z2))
            pos = torch.cat((pos1, pos2))
        else:
            x = x1
            y = y1
            z = z1
            pos = pos1          
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='normal'):
            labels = torch.tensor([self.B(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='reccirc'):
            labels = torch.tensor(self.reccircB(x, y, z), requires_grad=True)
        if(os.path.exists('./data/xyz.pt')==False):
            torch.save(pos, f'./data/xyz.pt')
        if(os.path.exists('./data/B.pt')==False):
            torch.save(labels, f'./data/B.pt')
        return  pos, labels
    
    def train_data_slice(self, Btype='normal'):
        L = self.L
        N = self.N_sample
        x = np.array([0,0,0,0,0,0]).reshape(-1,1)
        y = np.array([-L,0,L,-L,0,L]).reshape(-1,1)        
        z = np.array([L,L,L,-L,-L,-L]).reshape(-1,1)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)
        pos = torch.cat((x, y, z), axis=1)
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='normal'):
            labels = torch.tensor([self.B(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='reccirc'):

            labels = torch.tensor([self.reccircB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        if(os.path.exists('./data/xyz_slice.pt')==False):
            torch.save(pos, f'./data/xyz_slice.pt')
        if(os.path.exists('./data/B_slice.pt')==False):
            torch.save(labels, f'./data/B_slice.pt')
        return  pos, labels

    def test_data_cube(self, Btype='normal'):
        L = self.L
        N = self.N_test
        x = np.random.default_rng().uniform(low = -L, high = L, size = ((N, 1))) #shape： 1000*1
        y = np.random.default_rng().uniform(low = -L, high = L, size = ((N, 1)))
        z = np.random.default_rng().uniform(low = -L, high = L, size = ((N, 1)))
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)        
        pos = torch.cat((x, y, z), axis = 1)
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='normal'):
            labels = torch.tensor([self.B(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='reccirc'):
            labels = torch.tensor(self.reccircB(x, y, z), requires_grad=True)
        if(os.path.exists('./data/test_xyz.pt')==False):
            torch.save(pos, f'./data/test_xyz.pt')
        if(os.path.exists('./data/test_B.pt')==False):
            torch.save(labels, f'./data/test_B.pt')
        return pos, labels

    def test_data_slice(self, Btype='normal'):
        L = self.L
        N = int(self.N_test/10)
        x = np.random.default_rng().uniform(low = -L/10, high = L/10, size = ((N, 1)))
        y = np.random.default_rng().uniform(low = -L/2, high = L/2, size = ((N, 1))) #shape： 1000*1
        z = np.random.default_rng().uniform(low = -L/2, high = L/2, size = ((N, 1)))
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)
        pos = torch.cat((x, y, z), axis = 1)
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='normal'):
            labels = torch.tensor([self.B(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        elif(Btype=='reccirc'):
            labels = torch.tensor([self.reccircB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        if(os.path.exists('./data/test_xyz_slice.pt')==False):
            torch.save(pos, f'./data/test_xyz_slice.pt')
        if(os.path.exists('./data/test_B_slice.pt')==False):
            torch.save(labels, f'./data/test_B_slice.pt')
        return pos, labels
def sampling(train_data,train_labels,models,idx):
    if(idx>0):
        model=models.models[idx-1].to('cpu')
        prob=torch.sum((model(train_data)-train_labels[:,:,0])**2,dim=1)
        prob=prob.detach().numpy()
        prob=prob/np.sum(prob)
        random_index=np.random.choice(train_data.shape[0],train_data.shape[0],p=prob)
        train_data_np=train_data.detach().numpy()
        train_labels_np=train_labels.detach().numpy()
        train_data=np.concatenate((train_data_np[random_index,:].copy(),train_data_np.copy()),axis=0)
        train_labels=np.concatenate((train_labels_np[random_index,:].copy(),train_labels_np.copy()),axis=0)
        train_data=torch.tensor(train_data,dtype=torch.float32,requires_grad=True)
        train_labels=torch.tensor(train_labels,dtype=torch.float32,requires_grad=True)
    return train_data,train_labels
