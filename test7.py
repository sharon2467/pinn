#这个程序用于转换学长传来的磁场数据
import numpy as np
name='0'
train_set=np.load('磁场/中心磁场'+name+'/train_set.npy')
test_set=np.load('磁场/中心磁场'+name+'/test_set.npy')

train_set=np.concatenate((train_set[:,4:],train_set[:,1:4]),axis=1)
test_set=np.concatenate((test_set[:,4:],test_set[:,1:4]),axis=1)
print(train_set)
np.save('experiment'+name,np.concatenate((train_set,test_set),axis=0))