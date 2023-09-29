import os
while os.path.split(os.getcwd())[1] != "unrolling":
    os.chdir("..")
    if os.getcwd() == "/":
        raise ValueError()
print("Current working directory: {}".format(os.getcwd()))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
import pickle

import sys
PATH ='./codes/'
sys.path.insert(1,PATH)
import GLPALM as GLPALM 
import LPALM as LPALM 
import pytorch_lightning as pl

import argparse
parser = argparse.ArgumentParser(description="Training params")

parser.add_argument(
    "--Model", dest="model", default='GLPALM', help="GLPALM / LPALM"
)

parser.add_argument(
    "--Dataset", dest="dataset", default='mixture_sigma0.1', help="dataset of different variablity of amplitude: sigma0 / sigma0.01/ sigma0.1/ sigmainf"
)

parser.add_argument(
    "--var_Scaling", dest="sig_s", type=float, default=0., help="variation of the global scaling"
)

parser.add_argument(
    "--Batch_Size", dest="bs", type=int, default=64, help="Batch size"
)

parser.add_argument(
    "--learning_rate", dest="lr", type=float, default=1e-3, help="learning rate"
)

parser.add_argument(
    "--NEpochs", dest="nepochs", type=int, default=10000, help="number of epochs for training"
)

parser.add_argument(
    "--NLayers", dest="nlayers", type=int, default=2, help="number of layers",
)

parser.add_argument(
    "--lam_loss", dest="lam_loss", type=float, default=0., help="weight of amplitude-estimation error in the loss",
)

parser.add_argument(
    "--layers_weights", dest="layers_weights", default='last', help="type of loss control on intermediate layers: last, exp, ",
)

parser.add_argument(
    "--update_A", dest="update_A", default='LS', help="GD, LS, No-updating",
)

parser.add_argument(
    "--Version", dest="version", default=None, help="CNN",
)

parser.add_argument(
    "--W_shape", dest="W_shape", default='vector', help="W vector or matrix",
) # for LPALM 


args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
    kwargs = {}
    acc = "gpu"
    Xpus_per_trial = 1
else:
    device = 'cpu'
    kwargs = {}
    acc = 'cpu'
    Xpus_per_trial = 1

print("device USED: ",device)

if device == 'cuda': # if GPU
    torch.backends.cudnn.benchmark = True

if args.sig_s==0.:
    with open('data/'+args.dataset+'.npy', "rb") as f:
        Y, A, X = pickle.load(f)
else:
    with open('data/'+args.dataset+'_scaling'+str(args.sig_s)+'.npy', "rb") as f:
        Y, A, X = pickle.load(f)

idx_train = np.arange(4000)
idx_val = np.arange(4000,5000)
Xtrain = X[idx_train]
Ytrain = Y[idx_train]
Atrain = A[idx_train]
Xval = X[idx_val]
Yval = Y[idx_val]
Aval = A[idx_val]

train_dataloader = DataLoader(TensorDataset(torch.from_numpy(Ytrain).float(), torch.from_numpy(Atrain).float(), torch.from_numpy(Xtrain).float()), batch_size=args.bs)
val_dataloader = DataLoader(TensorDataset(torch.from_numpy(Yval).float(), torch.from_numpy(Aval).float(), torch.from_numpy(Xval).float()), batch_size=args.bs)

n = args.nlayers

if args.dataset == 'noisy_Ba133':
    l = ['Models/IAE/cnn_test']
elif args.dataset == 'noisy_Co57':
    l = ['Models/IAE/cnn_test2']
elif args.dataset == 'noisy_Cs137':
    l = ['Models/IAE/cnn_test3']
elif 'mixture' in args.dataset:
    l = ['Models/IAE/cnn_test', 'Models/IAE/cnn_test2', 'Models/IAE/cnn_test3']

#####################################################################################################################
# test with diferent weighted loss
#####################################################################################################################
if args.layers_weights == 'last': # no control on the intermediate layers
    layer_weights = torch.zeros(n)
    layer_weights[-1]=1. 
elif args.layers_weights == 'exponential': # exponential weights for loss on all layers
    layer_weights = 2**torch.arange(n) 
elif args.layers_weights == 'linear': # linear weights for loss on all layers
    layer_weights = torch.arange(n)+1 
elif args.layers_weights == 'uniform': # uniform weights for loss on all layers
    layer_weights = torch.ones(n) 
else:
    raise ValueError("argument --loss must be one of last, exponential, linear and uniform")

layer_weights = layer_weights/layer_weights.sum() #normalize weights

#####################################################################################################################
# trainer for GLPALM
#####################################################################################################################
if args.model =='GLPALM':
    arg_GLPALM = GLPALM.get_GLPALM_args(fname_list=l, NLayers=n, lam_loss=args.lam_loss, layers_weights=layer_weights, update_A=args.update_A, version=args.version, device=device)
    arg_train = {"learning_rate":args.lr}
    model = GLPALM.GLPALM(arg_GLPALM, arg_train).to(device)

    if args.update_A == 'No-updating':
        model_name = 'GLFBS'
    else:
        model_name = 'GLPALM_'+args.update_A
    if not args.version is None:
        model_name = model_name + '_' +args.version
    model_name = model_name +'_L'+str(n)+args.dataset[args.dataset.find('_'):]

    if args.sig_s != 0:
        model_name = model_name + '_scaling' + str(args.sig_s)

    if args.lam_loss != 0:
        model_name = model_name + '_ERec'+ str(args.lam_loss)
    if args.layers_weights != 'last':
        model_name = model_name + '_WL_' + args.layers_weights
    if os.path.isfile('./Models/model_'+ model_name+'.pth'): 
        model = GLPALM.load_model('./Models/model_'+model_name,device=device)
        model_name = model_name + '_retrained'
    print('model loaded: ', os.path.isfile('./Models/model_'+ model_name+'.pth'))  

    logger = TensorBoardLogger(save_dir='./CKPT', version=model_name, name="lightning_logs")
    trainer = pl.Trainer(callbacks=[], default_root_dir='./CKPT', max_epochs=args.nepochs, logger=logger)
    # print(trainer.__dict__)
    trainer.fit(model, train_dataloader, val_dataloader)
    save_name = './Models/model_'+ model_name
    GLPALM.save_model(model, save_name)


#####################################################################################################################
# trainer for LPALM
#####################################################################################################################
if args.model =='LPALM':

    if args.W_shape == 'matrix':
        arg_LPALM = LPALM.get_LPALM_args(fname_list=l, NLayers=n, update_A=args.update_A, W_diag=False, lam=args.lam_loss, device=device)
    elif args.W_shape == 'vector':
        arg_LPALM = LPALM.get_LPALM_args(fname_list=l, NLayers=n, update_A=args.update_A, W_diag=True, lam=args.lam_loss, device=device)
    else:
        print('Unrecognized shape of W')
    
    arg_train = {"learning_rate":args.lr}
    model = LPALM.LPALM(arg_LPALM, arg_train).to(device)

    if args.update_A == 'No-updating':
        model_name = 'LFBS'
    else:
        model_name = 'LPALM_'+args.update_A

    model_name = model_name +'_L'+str(n)+args.dataset[args.dataset.find('_'):]

    if args.sig_s != 0:
        model_name = model_name + '_scaling' + str(args.sig_s)

    if args.lam_loss != 0:
        model_name = model_name + '_ERec'+ str(args.lam_loss)
    if args.layers_weights != 'last':
        model_name = model_name + '_WL_' + args.layers_weights
    if os.path.isfile('./Models/model_'+ model_name+'.pth'): 
        model = LPALM.load_model('./Models/model_'+model_name,device=device)
        model_name = model_name + '_retrained'
    print('model loaded: ', os.path.isfile('./Models/model_'+ model_name+'.pth'))  

    logger = TensorBoardLogger(save_dir='./CKPT', version=model_name, name="lightning_logs")
    trainer = pl.Trainer(callbacks=[], default_root_dir='./CKPT', max_epochs=args.nepochs, logger=logger)
    # print(trainer.__dict__)
    trainer.fit(model, train_dataloader, val_dataloader)
    save_name = './Models/model_'+ model_name
    LPALM.save_model(model, save_name)