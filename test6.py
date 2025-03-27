import numpy as np
train_data=np.load('train_set.npy')
test_data=np.load('test_set.npy')
train_data=np.concatenate((train_data[:,3:],train_data[:,:3]),axis=1)
test_data=np.concatenate((test_data[:,3:],test_data[:,:3]),axis=1)
experiment2=np.concatenate((train_data,test_data),axis=0)
np.save('experiment2.npy',experiment2)