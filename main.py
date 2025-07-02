from data import *
from model import *
from train import *
from Eval import *
from utils import *
import argparse
import json
# 这个函数用于标准化训练和测试数据以及标签，采用的方法是平均值为0方差为1的标准化。只有在config['standard']==1时才会进行磁场的标准化。
# 只有在config['data_standard']==1时才会进行坐标的标准化。
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

# 这部分用于定义输入的参数组，当发起一次程序运行时，格式应当为：python main.py --vars=value train/import/eval --vars=value
parser = argparse.ArgumentParser(description='PINN field prediction',exit_on_error=True,allow_abbrev=False)
subparser=parser.add_subparsers(dest='mode', help='three modes ')
subparser_train=subparser.add_parser('train', help='train the model from simulation data')
subparser_import=subparser.add_parser('import', help='train the model from experimental data')
subparser_eval=subparser.add_parser('eval', help='eval the trained model')
subparser_train_Btype=subparser_train.add_subparsers(dest='Btype', help='type of field for simulation')
subparser_train_Helmholtz=subparser_train_Btype.add_parser('Helmholtz', help='two head to head helmholtz coils')
subparser_train_normal=subparser_train_Btype.add_parser('normal', help='three pairs of coils in xyz directions')
subparser_train_reccirc=subparser_train_Btype.add_parser('reccirc', help='two pairs of circular coils,and one pair of rectangular coils')
parser.add_argument('--seed', type=int, default=42, metavar='--',
                    help='random seed')
parser.add_argument('--model_mode', type=str, metavar='--', choices=['phi','B','hard'],default='B',help='B means model predicts the field,phi means model predicts magnetic scalar potential,hard means model predicts the scalar potential with no boundary loss,but due to statistical instalbility is failed')

parser.add_argument('--logdir', type=str, default='./log/', metavar='--',
                    help='log dir,no need to change')
parser.add_argument('--layers', type=int, default=4, metavar='--',
                    help='number of layers in the network')
parser.add_argument('--experiment', type=str, default='training', metavar='--',
                    help='name of the experiment you want to do, like scan different learning rate, scan different sample points')
parser.add_argument('--device', type=str, default='cpu', metavar='--', choices=['cpu', 'cuda:0'],
                    help='device type, cpu or cuda:0')
parser.add_argument('--lr', type=float, default=0.001, metavar='--',
                    help='learning rate')
parser.add_argument('--adjust_lr', type=int, default=0, metavar='--', choices=[0, 1],
                    help='whether adjust the lr during training, 0 means no, 1 means yes')

parser.add_argument('--units', type=int, default=32, metavar='--', 
                    help='number of neurals in a network layer')
parser.add_argument('--Nep', type=int, default=100001, metavar='--', 
                    help='number of epochs')
parser.add_argument('--Npde', type=int, default=256, metavar='--',
                    help='number of points to join the PDE calculation')
parser.add_argument('--addBC', type=int, default=0, metavar='--', choices=[0, 1],
                    help='add BC constrains or not, 0 means no, 1 means yes')
parser.add_argument('--standard', type=int, default=0, metavar='--', choices=[0, 1],
                    help='perform standardization on labels or not, 0 means no, 1 mean yes')
parser.add_argument('--Lambda',type=float,default=1,metavar='--',help='super variable,loss=data_loss+Lambda*PDE_loss')
parser.add_argument('--N_models',type=int,metavar='--',help='number of models to train,this program supports prediction of multiple models,user can choose different combination method when evaluating',default=1)
parser.add_argument('--data_standard',type=int,default=0,metavar='--',choices=[0,1],help='perform standardization on data or not')
subparser_import.add_argument('--eval_path',type=str,metavar='--',help='path to the data you want to evaluate')
subparser_eval.add_argument('--model_path',type=str,metavar='--',help='path to the model you want to evaluate')
subparser_eval.add_argument('--data_type',type=str,default='experimental',metavar='--',choices=['experimental','simulation'],help='data type of the model you want to evaluate, experimental means the model is trained with experimental data, simulation means the model is trained with simulation data')
subparser_eval.add_argument('--data_path',type=str,metavar='--',help='path to the data you want to evaluate, if data_type is experimental, this should be the path to the experimental data')
subparser_train.add_argument('--train_sampling',type=str,metavar='--',default='uniform',choices=['linspace','uniform','normal'],help='how to sample train point from cube surface in simulation,only available on simulation mode, linspace means evenly spaced points, uniform means uniformly distributed points, normal means normally distributed points')
subparser_train.add_argument('--random_sample',type=int,metavar='--',default=0,choices=[0,1],help='random sample the train data or not,if yes,every model will be trained with different train data selected base on previous models performance,only available on simulation mode,improvement not observed')
subparser_train.add_argument('--length', type=float, default=1, metavar='--',
                    help='side length of the area that you want to predict,only used in simulation mode,default is 1')

subparser_train.add_argument('--Nsamples', type=int, default=16, metavar='--',
                    help='number of sample points per surface, only used in simulation mode,default is 16')
subparser_train.add_argument('--Ntest', type=int, default=1000, metavar='--', 
                    help='number of test points, only used in simulation mode,default is 1000')
subparser_train.add_argument('--geo', type=str, default='cube', metavar='--', choices=['cube', 'slice'],
                    help='geo of the coils, cube means fields are measured in a cube, slice means fields are measured in a slice')
# 接下来的所有参数都只用于模拟模式，用于控制模拟数据生成器采用的线圈模型的几何参数。
subparser_train.add_argument('--radius', type=float, default=1, metavar='--',
                    help='radius of the coils,only use in simulation mode,default is 1')
subparser_train.add_argument('--inner_sample', type=int, default=0, metavar='--', choices=[1,0],help='whether sample the inner part of the cube or not')
subparser_train.add_argument('--dx', type=float, default=9999, metavar='--',help='the distance in x direction of the two helmholtz coils')
subparser_train.add_argument('--dy', type=float, default=9999, metavar='--',help='the distance in y direction of the two helmholtz coils')
subparser_train.add_argument('--dz', type=float, default=9999, metavar='--',help='the distance in z direction of the two helmholtz coils')
subparser_train_reccirc.add_argument('--radius1', type=float, default=9999, metavar='--',help='the radius of the first helmholtz coil')
subparser_train_reccirc.add_argument('--radius2', type=float, default=9999, metavar='--',help='the radius of the second helmholtz coil')
subparser_train_reccirc.add_argument('--a', type=float, default=9999, metavar='--',help='x length of the rectangle')
subparser_train_reccirc.add_argument('--b', type=float, default=9999, metavar='--',help='y length of the rectangle')
subparser_train_reccirc.add_argument('--Iz', type=float, default=9999, metavar='--',help='z Intensity only used in reccirc')
subparser_train_reccirc.add_argument('--Ix', type=float, default=9999, metavar='--',help='x Intensity only used in reccirc')
subparser_train_reccirc.add_argument('--Iy', type=float, default=9999, metavar='--',help='y Intensity only used in reccirc')

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
config = {}
config.update(vars(args))
print(config)
if(args.mode=='import'):
    # 这是利用实验数据进行训练并评估的模式
    # 这段专用于导入实验数据，实验数据应当是一个numpy数组，前98行是训练数据，后面是测试数据。数据的前三列是坐标，后三列是磁场分量。
    config['logdir'] = args.logdir + '/' + args.experiment
    path = mkdir(config['logdir'])
    config['path'] = path
    N_models=config['N_models']
    temp=np.load(f"{args.eval_path}.npy")
    config['length']=np.max(temp[:,:3])-np.min(temp[:,:3])
    train_data=torch.tensor(temp[:98,:3],dtype=torch.float32)
    train_labels=torch.tensor(temp[:98,3:],dtype=torch.float32)
    test_data=torch.tensor(temp[98:,:3],dtype=torch.float32)
    test_labels=torch.tensor(temp[98:,3:],dtype=torch.float32)
    # 打印分割后的数据形状
    print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
    # 对数据进行标准化处理
    train_data1,train_labels1,test_data1,test_labels1,config=standardization(train_data,train_labels,test_data,test_labels,config)
    # 保存配置，训练集和测试集
    with open(f"{path}/config.json", 'w') as config_file:
        config_file.write( json.dumps(config, indent=4) )
    np.save(f"{path}/train_data.npy", train_data)
    np.save(f"{path}/train_labels.npy", train_labels)
    #初始化模型组，一个模型组内含多个子模型，通过模型组的实例化程序得以进行集成预测
    models=MODELS(config,train_data,train_labels)
    #分别训练每个模型
    for i in range(N_models):
        model = train( train_data1, train_labels1, test_data1, test_labels1, config,i )
        models.models.append(model)
        print(model.state_dict())
    #models.load('./log/B400mediummodel32-2-Npde1000/2025_6_30_20_44_3')
    #对模型进行评估
    Eval(models,config,(test_data,test_labels),args.mode)
if(args.mode=='train'):
    # 若未设定模拟模式中三个方向上的距离，则自动设为两倍半径的距离。
    if((args.dx==9999) and (args.dy==9999) and (args.dz==9999)):
        args.dx = args.radius*2
        args.dy = args.radius*2
        args.dz = args.radius*2
    if(args.Btype=='reccirc'):
        if((args.radius1==9999) and (args.radius2==9999)):
            args.radius1 = args.radius
            args.radius2 = args.radius
    #这是利用模拟数据进行训练并评估的模式
    config['logdir']    = args.logdir + '/' + args.experiment
    path = mkdir(config['logdir'])
    config['path'] = path
    #数据生成器同样是个实例
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
    #生成数据
    if(config['geo']=='cube'):
        train_data, train_labels = field.train_data_cube(config['Btype'],config['inner_sample'],config['train_sampling'])
        test_data, test_labels = field.test_data_cube(config['Btype'])
    if(config['geo']=='slice'):
        train_data, train_labels = field.train_data_slice(config['Btype'])
        test_data, test_labels = field.test_data_slice(config['Btype'])
    N_models=config['N_models']
    # 对数据进行标准化处理
    train_data,train_labels,test_data,test_labels,config=standardization(train_data,train_labels,test_data,test_labels,config)
    print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")  
    with open(f"{path}/config.json", 'w') as config_file:
        config_file.write( json.dumps(config, indent=4) )
    np.save(f"{path}/train_data.npy", train_data)
    np.save(f"{path}/train_labels.npy", train_labels)
    models=MODELS(config,train_data,train_labels)
    # 如果开启random_sample，则每个模型将会从训练数据中按照前一个模型的表现随机采样一部分数据进行训练。
    for i in range(N_models):
        if(config['random_sample']==1):
            train_data1,train_labels1=sampling(train_data,train_labels,models,i)
            model = train( train_data1, train_labels1, test_data, test_labels, config,i ) 
        else:
            model = train( train_data, train_labels, test_data, test_labels, config,i )
        models.models.append(model)
    #评估模型
    Eval(models,config,field,args.mode)
if(args.mode=='eval'):
    # 这是利用已经训练好的模拟数据模型进行评估的模式
    if(args.data_type=='simulation'):
        with open(f"{args.model_path}/config.json", 'r') as config_file:
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
        train_data = torch.tensor(np.load(f"{args.model_path}/train_data.npy"))
        train_labels = torch.tensor(np.load(f"{args.model_path}/train_labels.npy"))
        models=MODELS(config,train_data,train_labels)
        models.load(args.model_path)
        Eval(models,config,field,args.mode)
    # 这是利用已经训练好的实验数据模型进行评估的模式
    if(args.data_type=='experimental'):
        config['logdir'] = args.logdir + '/' + args.experiment
        path = mkdir(config['logdir'])
        config['path'] = path
        N_models=config['N_models']
        temp=np.load(f"{args.data_path}.npy")
        config['length']=np.max(temp[:,:3])-np.min(temp[:,:3])
        train_data=torch.tensor(temp[:98,:3],dtype=torch.float32)
        train_labels=torch.tensor(temp[:98,3:],dtype=torch.float32)
        test_data=torch.tensor(temp[98:,:3],dtype=torch.float32)
        test_labels=torch.tensor(temp[98:,3:],dtype=torch.float32)
    # 打印分割后的数据形状
        print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
    # 对数据进行标准化处理
        train_data1,train_labels1,test_data1,test_labels1,config=standardization(train_data,train_labels,test_data,test_labels,config)
    # 保存配置，训练集和测试集
        with open(f"{path}/config.json", 'w') as config_file:
            config_file.write( json.dumps(config, indent=4) )
        np.save(f"{path}/train_data.npy", train_data)
        np.save(f"{path}/train_labels.npy", train_labels)
        #初始化模型组，一个模型组内含多个子模型，通过模型组的实例化程序得以进行集成预测
        models=MODELS(config,train_data,train_labels)
        models.load(args.model_path)
        print(models.models[0].state_dict())
    #对模型进行评估
    Eval(models,config,(test_data,test_labels),args.data_type)
