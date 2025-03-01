import torch
import numpy as np
from model import *
import time
import matplotlib.pyplot as plt

def lr_adjust(val_loss, optimizer):
    if(val_loss<0.1):
        for g in optimizer.param_groups:
            if(g['lr']>0.001):
                g['lr'] = 0.0001
    if(val_loss<0.01):
        for g in optimizer.param_groups:
            if(g['lr']>0.0001):
                g['lr'] = 0.00001
    if(val_loss<0.001):
        for g in optimizer.param_groups:
            if(g['lr']>0.00001):
                g['lr'] = 0.000001

def train(train_data, train_labels, test_data, test_labels, config,num):
    Nep    = config['Nep']
    units  = config['units']
    device = config['device']
    lr     = config['lr']
    L      = config['length']
    path   = config['path'] 
    Npde   = config['Npde']
    adjust = config['adjust_lr']
    addBC  = config['addBC']
    Lambda = config['Lambda']
    model  = PINN(units,config['model_mode'])
    model.to(device)
    #model.load_state_dict(torch.load('best_model.pt'))
    optimizer1 = optim.AdamW(model.parameters(), lr)
    optimizer2=optim.LBFGS(model.parameters(),lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer1,patience=800)
    criterion = PINN_Loss(Npde, L, device, addBC,Lambda)
    
    loss_f_l = []
    loss_u_l = []
    loss_cross_l = []
    loss_BC_div_l = []
    loss_BC_cul_l = []
    loss_l = []
    test_loss_l = []
    epoch = []

    mini_loss = 100000000
    best_model = model
    best_ep = 0
    train_data = train_data.requires_grad_(True).to(device)
    train_labels = train_labels.requires_grad_(True).to(device)
    test_data=test_data.to(device)
    test_labels=test_labels.to(device)     
    st = time.time()
    exitflag=0
    patience=100
    batch_on=False
    def closure():
        optimizer2.zero_grad()
        pred = model(train_data_batch)
        loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion(train_data_batch, pred, train_labels_batch, model)
        loss.backward(retain_graph=True)
        return loss
    for ep in range(Nep):
        model.train()
        batch_size=128
        if(batch_on):
            r=torch.randint(train_data.shape[0],(batch_size,))
            train_data_batch=train_data[r,:]
            train_labels_batch=train_labels[r,:]
        else:
            train_data_batch=train_data
            train_labels_batch=train_labels
        if(ep<Nep*0.95 and exitflag<patience):
            optimizer1.zero_grad()            
            pred = model(train_data_batch)      
            loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion(train_data_batch, pred, train_labels_batch, model)
            loss.backward()
            optimizer1.step()
            if(adjust):
                scheduler.step(loss)
        else:
            optimizer2.step(closure=closure)
            pred = model(train_data_batch)
            loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion(train_data_batch, pred, train_labels_batch, model)
        if(ep%100==0):
            epoch.append(ep)
            loss_f_l.append(loss_f.item())
            loss_u_l.append(loss_u.item())
            loss_cross_l.append(loss_cross.item())
            loss_BC_div_l.append(loss_BC_div.item())
            loss_BC_cul_l.append(loss_BC_cul.item())
            loss_l.append(loss.item())
            model.eval()
            test_pred = model(test_data)

            test_loss = torch.mean(torch.square(test_pred-test_labels))

            test_loss_l.append(test_loss.item())

                #if(adjust):
                    #lr_adjust(test_loss, optimizer1)            
            if(mini_loss>test_loss and exitflag<patience):
                torch.save(model.state_dict(), f'{path}/best_model{num}.pt')
                mini_loss = test_loss
                best_model = model
                best_ep = ep
                exitflag=0
            else:
                exitflag=exitflag+1

            if(test_loss<0.0000001 or exitflag>patience+5):
                print('early stop!!!')
                break
            print(f'===>>> ep: {ep}')
            print(f'time used: {time.time()-st:.2f}s, time left: {(time.time()-st)/(ep+1)*Nep-(time.time()-st):.2f}s')
            print(f'loss_B: {loss_u:.7f}, loss_div: {loss_f:.7f}, loss_cul: {loss_cross:.7f}, loss_BC_div: {loss_BC_div:.7f}, loss_BC_cul: {loss_BC_cul:.7f}')
            print(f'total loss: {loss:.7f}, test loss: {test_loss:.7f}')
            #print(torch.abs(test_pred-test_labels/test_labels))
            print(f'max{torch.max(torch.abs(test_pred-test_labels))}')
    print(f'best loss at ep: {best_ep}, best_loss: {mini_loss:.7f}')
    print(f'total time used: {time.time()-st:.2f}s')
    plt.plot(epoch, loss_f_l, label='loss div')
    plt.plot(epoch, loss_u_l, label='loss B')
    plt.plot(epoch, loss_cross_l, label='loss cul')
    plt.plot(epoch, loss_BC_div_l, label='loss BC div')
    plt.plot(epoch, loss_BC_cul_l, label='loss BC cul')    
    plt.plot(epoch, loss_l, label='total loss')
    plt.plot(epoch, test_loss_l, label='test loss')
    plt.scatter(best_ep, mini_loss.to('cpu').item(), label='test best loss', marker='*')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{path}/loss'+str(num)+'.png')  
    plt.show()
    plt.close()
    
    np.save(f'{path}/loss_div'+str(num)+'.npy',      np.array(loss_f_l))
    np.save(f'{path}/loss_B'+str(num)+'.npy',    np.array(loss_u_l))
    np.save(f'{path}/loss_cul'+str(num)+'.npy',    np.array(loss_cross_l))
    np.save(f'{path}/loss_BC_div'+str(num)+'.npy', np.array(loss_BC_div_l))
    np.save(f'{path}/loss_BC_cul'+str(num)+'.npy', np.array(loss_BC_cul_l))
    np.save(f'{path}/loss'+str(num)+'.npy',        np.array(loss_l))
    np.save(f'{path}/loss_test'+str(num)+'.npy',   np.array(test_loss_l))

    return best_model
