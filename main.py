from data import *
from model import *
from train import *
from Eval import *
from utils import *
import argparse
import json
from sklearn.model_selection import train_test_split
def standardization(train_data,train_labels,test_data,test_labels,config):
    if(config['standard']==1):
        mean = torch.mean(train_labels,0)
        std  = torch.std(train_labels,0)
        train_labels = ((train_labels - mean)/std).detach().numpy()
        test_labels  = ((test_labels  - mean)/std).detach().numpy()
        train_labels = torch.tensor(train_labels)
        test_labels  = torch.tensor(test_labels)
        config['mean'] = mean.detach().numpy().tolist()
        config['std']  = std.detach().numpy().tolist()
    else:
        config['mean'] = 0
        config['std']  = 1
    if(config['data_standard']==1):
        mean = torch.mean(train_data,0)
        std  = torch.std(train_data,0)
        train_data = ((train_data - mean)/std).detach().numpy()
        test_data  = ((test_data  - mean)/std).detach().numpy()
        train_data = torch.tensor(train_data)
        test_data  = torch.tensor(test_data)
        config['mean_data'] = mean.detach().numpy().tolist()
        config['std_data']  = std.detach().numpy().tolist()
    else:
        config['mean_data'] = 0
        config['std_data']  = 1
    return train_data,train_labels,test_data,test_labels,config
parser = argparse.ArgumentParser(description='PINN field ediction')
parser.add_argument('--model_mode', type=str, metavar='--', choices=['phi','B','hard'],default='B',)
parser.add_argument('--mode', type=str, metavar='--', choices=['train','import', 'eval'],default='train',)
parser.add_argument('--eval_path',type=str,metavar='--',help='path to the model you want to evaluate')
parser.add_argument('--logdir', type=str, default='./log/', metavar='--',
                    help='log dir')
parser.add_argument('--experiment', type=str, default='training', metavar='--',
                    help='name of the experiment you want to do, like scan different learning rate, scan different sample points')
parser.add_argument('--device', type=str, default='cpu', metavar='--', choices=['cpu', 'cuda:0'],
                    help='device type, cpu or cuda:0')
parser.add_argument('--lr', type=float, default=0.001, metavar='--',
                    help='learning rate')
parser.add_argument('--adjust_lr', type=int, default=0, metavar='--', choices=[0, 1],
                    help='whether adjust the lr during training, 0 means no, 1 means yes')
parser.add_argument('--Nsamples', type=int, default=16, metavar='--',
                    help='number of sample points per surface')
parser.add_argument('--Ntest', type=int, default=1000, metavar='--', 
                    help='number of test points')
parser.add_argument('--radius', type=float, default=1, metavar='--',
                    help='radius of the coils')
parser.add_argument('--length', type=float, default=1, metavar='--',
                    help='side length of the area that you want to predict')
parser.add_argument('--units', type=int, default=32, metavar='--', 
                    help='number of neurals in a network layer')
parser.add_argument('--Nep', type=int, default=100001, metavar='--', 
                    help='number of epochs')
parser.add_argument('--Npde', type=int, default=256, metavar='--',
                    help='number of points to join the PDE calculation')
parser.add_argument('--addBC', type=int, default=0, metavar='--', choices=[0, 1],
                    help='add BC constrains or not, 0 means no, 1 means yes')
parser.add_argument('--standard', type=int, default=0, metavar='--', choices=[0, 1],
                    help='perform standardization or not, 0 means no, 1 mean yes')
parser.add_argument('--data_standard',type=int,default=0,metavar='--',choices=[0,1],help='perform standardization on data or not')
parser.add_argument('--geo', type=str, default='cube', metavar='--', choices=['cube', 'slice'],
                    help='geo of the coils')
parser.add_argument('--inner_sample', type=int, default=0, metavar='--', choices=[1,0],help='whether sample the inner part of the cube or not')
parser.add_argument('--Btype', type=str, default='Helmholtz', metavar='--', choices=['Helmholtz', 'normal','reccirc'],
                    help='which type field you want to generate, Helmholtz or normal field')
parser.add_argument('--dx', type=float, default=9999, metavar='--',help='the distance in x direction of the two helmholtz coils')
parser.add_argument('--dy', type=float, default=9999, metavar='--',help='the distance in y direction of the two helmholtz coils')
parser.add_argument('--dz', type=float, default=9999, metavar='--',help='the distance in z direction of the two helmholtz coils')
parser.add_argument('--radius1', type=float, default=9999, metavar='--',help='the radius of the first helmholtz coil')
parser.add_argument('--radius2', type=float, default=9999, metavar='--',help='the radius of the second helmholtz coil')
parser.add_argument('--a', type=float, default=9999, metavar='--',help='x length of the rectangle')
parser.add_argument('--b', type=float, default=9999, metavar='--',help='y length of the rectangle')
parser.add_argument('--Iz', type=float, default=9999, metavar='--',help='z Intensity only used in reccirc')
parser.add_argument('--Ix', type=float, default=9999, metavar='--',help='x Intensity only used in reccirc')
parser.add_argument('--Iy', type=float, default=9999, metavar='--',help='y Intensity only used in reccirc')
parser.add_argument('--Lambda',type=float,default=1,metavar='--',help='super variable')
parser.add_argument('--N_models',type=int,metavar='--',help='number of models to train',default=1)
parser.add_argument('--train_sampling',type=str,metavar='--',default='uniform',choices=['linspace','uniform','normal'],help='train point sampling mode')
parser.add_argument('--random_sample',type=int,metavar='--',default=0,choices=[0,1],help='random sample the train data or not')
parser.add_argument('--noise_data',type=float,metavar='--',default=0,help='noise on train data')
parser.add_argument('--noise_labels',type=float,metavar='--',default=0,help='noise on train labels')
args = parser.parse_args()
if((args.dx==9999) and (args.dy==9999) and (args.dz==9999)):
    args.dx = args.radius*2
    args.dy = args.radius*2
    args.dz = args.radius*2
if((args.radius1==9999) and (args.radius2==9999)):
    args.radius1 = args.radius
    args.radius2 = args.radius
config = {}
config.update(vars(args))
if(args.mode=='import'):
    config['logdir'] = args.logdir + '/' + args.experiment
    path = mkdir(config['logdir'])
    config['path'] = path
    N_models=config['N_models']
    temp=np.load(f"{args.eval_path}.npy")
    config['length']=np.max(temp[:,:3])-np.min(temp[:,:3])
    # 分割数据集，80%作为训练集，20%作为测试集
    #train_data, test_data, train_labels,test_labels = train_test_split(train_data_np, train_labels_np, test_size=0.1, random_state=42)  

    train_data=torch.tensor(temp[int(np.round(temp.shape[0]*0.15)):,:3],dtype=torch.float32)
    train_labels=torch.tensor(temp[int(np.round(temp.shape[0]*0.15)):,3:],dtype=torch.float32)
    test_data=torch.tensor(temp[0:int(np.round(temp.shape[0]*0.2)),:3],dtype=torch.float32)
    test_labels=torch.tensor(temp[0:int(np.round(temp.shape[0]*0.2)),3:],dtype=torch.float32)

    # 打印分割后的数据形状
    print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
    train_data1,train_labels1,test_data1,test_labels1,config=standardization(train_data,train_labels,test_data,test_labels,config)
    with open(f"{path}/config.json", 'w') as config_file:
        config_file.write( json.dumps(config, indent=4) )
    np.save(f"{path}/train_data.npy", train_data)
    np.save(f"{path}/train_labels.npy", train_labels)
    models=MODELS(config,train_data,train_labels)
    for i in range(N_models):
        model = train( train_data1, train_labels1, test_data1, test_labels1, config,i )
        models.models.append(model)
    Eval(models,config,(test_data,test_labels),args.mode)
if(args.mode=='train'):
    
    config['logdir']    = args.logdir + '/' + args.experiment
    path = mkdir(config['logdir'])
    config['path'] = path
    field = data_generation(radius=config['radius'],
                            N_sample=config['Nsamples'], 
                            N_test=config['Ntest'], 
                            L=config['length']/2,
                            dx=config['dx'],
                            dy=config['dy'],
                            dz=config['dz'],
                            radius1=config['radius1'],
                            radius2=config['radius2'],
                            a=config['a'],
                            b=config['b'],
                            Iz=config['Iz'],
                            Ix=config['Ix'],
                            Iy=config['Iy']
                        )
    if(config['geo']=='cube'):
        train_data, train_labels = field.train_data_cube(config['Btype'],config['inner_sample'],config['train_sampling'])
        test_data, test_labels = field.test_data_cube(config['Btype'])
    if(config['geo']=='slice'):
        train_data, train_labels = field.train_data_slice(config['Btype'])
        test_data, test_labels = field.test_data_slice(config['Btype'])
    N_models=config['N_models']
    #train_data=train_data+np.random.normal(0,1,np.size(train_data))*config['noise data']
    #train_labels=train_labels+np.random.normal(0,1,np.size(train_labels))*config['noise label']
    train_data,train_labels,test_data,test_labels,config=standardization(train_data,train_labels,test_data,test_labels,config)
    print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")  
    with open(f"{path}/config.json", 'w') as config_file:
        config_file.write( json.dumps(config, indent=4) )
    np.save(f"{path}/train_data.npy", train_data)
    np.save(f"{path}/train_labels.npy", train_labels)
    models=MODELS(config,train_data,train_labels)
    for i in range(N_models):
        if(config['random_sample']==1):
            train_data1,train_labels1=sampling(train_data,train_labels,models,i)
            model = train( train_data1, train_labels1, test_data, test_labels, config,i ) 
        else:
            model = train( train_data, train_labels, test_data, test_labels, config,i )
        models.models.append(model)
    Eval(models,config,field,args.mode)
if(args.mode=='eval'):
    with open(f"{args.eval_path}/config.json", 'r') as config_file:
        config = json.load(config_file)
    field = data_generation(radius=config['radius'],
                            N_sample=config['Nsamples'], 
                            N_test=config['Ntest'], 
                            L=config['length']/2,
                            dx=config['dx'],
                            dy=config['dy'],
                            dz=config['dz'],
                            radius1=config['radius1'],
                            radius2=config['radius2'],
                            a=config['a'],
                            b=config['b'],
                            Iz=config['Iz'],
                            Ix=config['Ix'],
                            Iy=config['Iy']
                        )
    train_data = torch.tensor(np.load(f"{args.eval_path}/train_data.npy"))
    train_labels = torch.tensor(np.load(f"{args.eval_path}/train_labels.npy"))
    models=MODELS(config,train_data,train_labels)
    models.load(args.eval_path)
    Eval(models,config,field,args.mode)

