import torch
import numpy as np
import IAE_CNN_TORCH_v2 as cnn
from utils import projection_simplex_torch
import pytorch_lightning as pl
import time

############################################################
# Saving model from dict
############################################################

def save_model(model,fname='model_GLPALM'+str(time.time())):

    params = {"arg_train":model.arg_train, "iae_model_fname_list": model.iae_fname_list, "NLayers": model.NLayers, "lam_loss": model.lam_loss, "layers_weights": model.layers_weights, 'update_A': model.update_A, 'version': model.version}
    
    torch.save({"model":model.state_dict(),"params":params}, fname+".pth")

############################################################
# Loading model from dict
############################################################

def load_model(fname,device="cpu"):

    model_in = torch.load(fname+".pth", map_location=device)
    params = model_in["params"]
    params["device"] = device
    model_state = model_in["model"]
    model = GLPALM(input_arg=params)
    model.load_state_dict(model_state)

    if device=='cuda':
        model = model.cuda()

    return model

############################################################
# GLPALM args
############################################################

def get_GLPALM_args(fname_list = None, NLayers=10, update_A='LS', device='cpu', learning_rate=1e-3, lam_loss=0., layers_weights = None, version = None):
    if layers_weights is None:
        layers_weights = torch.hstack([torch.zeros(NLayers-1),torch.tensor(1.)])   
    return {"iae_model_fname_list": fname_list, "NLayers": NLayers, "learning_rate":learning_rate, "lam_loss": lam_loss, 'layers_weights':layers_weights, 'update_A': update_A, 'version': version, 'device':device}

############################################################
# Main code
############################################################

class GLPALM(pl.LightningModule):
    '''
    iae_fname_list: IAE models corresponding aux sources to unmix
    NLayers: number of layers
    lam_loss: weight of A-related term in the loss
    layers_weights: weights of loss on the output of each layer
    update_A: different methods to update A ('GD':gradient descent, 'LS':least-squares, 'No-updating':no update for A -> GLFBS)
    version: different methods to substitute Jacobian Matrix in the update of lambda (MLP,CNN,CNN_v2)
    '''
    def __init__(self, input_arg=None, arg_train=None):
        super(GLPALM,self).__init__()
        torch.set_default_tensor_type(torch.FloatTensor)
        
        if input_arg is None:
            print("Run the get_arg first")
        
        self.iae_fname_list = input_arg["iae_model_fname_list"]   
        self.nb_sources = len(self.iae_fname_list)
        self.modelIAE_list = [cnn.load_model(fname, device=input_arg['device']) for fname in self.iae_fname_list]

        self.Lin = self.modelIAE_list[0].Lin
        self.num_ap = self.modelIAE_list[0].num_ap

        # check parameters of IAE models
        for iae in self.modelIAE_list:
            if not iae.mean_lambda:
                print('All layers of IAE models should have shared lambda !')                              
            if iae.num_ap != self.num_ap:
                print('All IAE models should have the same number of anchor points !')
            if iae.Lin != self.Lin:
                print('Anchor points of all IAE model should have the same length !')
            for param in iae.parameters():
                    param.requires_grad = False

        self.NLayers = input_arg["NLayers"]

        self.arg_train=arg_train
        self.lr=self.arg_train["learning_rate"]


        if 'lam_loss' in input_arg:
            self.lam_loss = input_arg['lam_loss'] # weight of estimation error of a in the loss
        else:
            self.lam_loss = 0.

        if 'layers_weights' in input_arg:
            self.layers_weights = input_arg['layers_weights'].to(input_arg['device'])
        else:
            self.layers_weights = torch.zeros(self.NLayers, device=input_arg['device'])
            self.layers_weights[-1] = 1.            

        if 'version' in input_arg:
            self.version = input_arg['version'] # weight of estimation error of a in the loss
        else:
            self.version = None

        if self.version == 'CNN': # CNN to substitue Jacobian Matrix
            for r in range(self.NLayers):
                for j in range(self.nb_sources):
                    ch = self.modelIAE_list[j].anchorpoints.shape[2]
                    W_conv = []
                    W_conv.append(torch.nn.Conv1d(ch, ch, kernel_size=7, stride=4, bias=False))
                    W_conv.append(torch.nn.Conv1d(ch, ch, kernel_size=7, stride=4, bias=False))
                    W_conv.append(torch.nn.Conv1d(ch, ch, kernel_size=7, stride=4, bias=False))
                    W_conv.append(torch.nn.Linear(5, 2, bias=False))
                    setattr(self,'W'+str(j+1)+'_l'+str(r+1),torch.nn.Sequential(*W_conv))

        elif self.version == 'CNN_v2': #CNN with non-linearity 
            for r in range(self.NLayers):
                for j in range(self.nb_sources):
                    ch = self.modelIAE_list[j].anchorpoints.shape[2]
                    W_conv = []
                    W_conv.append(torch.nn.Conv1d(ch, ch, kernel_size=7, stride=4, bias=False))
                    W_conv.append(torch.nn.BatchNorm1d(ch))
                    W_conv.append(torch.nn.ELU())
                    W_conv.append(torch.nn.Conv1d(ch, ch, kernel_size=7, stride=4, bias=False))
                    W_conv.append(torch.nn.BatchNorm1d(ch))
                    W_conv.append(torch.nn.ELU())
                    W_conv.append(torch.nn.Conv1d(ch, ch, kernel_size=7, stride=4, bias=False))
                    W_conv.append(torch.nn.BatchNorm1d(ch))
                    W_conv.append(torch.nn.ELU())
                    W_conv.append(torch.nn.Linear(5, 2, bias=False))
                    setattr(self,'W'+str(j+1)+'_l'+str(r+1),torch.nn.Sequential(*W_conv))

        else: # MLP to substitue Jacobian Matrix
            self.W = torch.zeros(self.nb_sources, self.num_ap, self.Lin)
            self.W = self.W.repeat(self.NLayers, 1, 1, 1)
            self.W = torch.nn.Parameter(self.W, requires_grad=True)


        if 'update_A' in input_arg:
            self.update_A = input_arg['update_A']
        else:
            self.update_A = 'LS'

        if self.update_A == 'GD':
            self.beta = torch.tensor(0.)
            self.beta = self.beta.repeat(self.NLayers, 1)
            self.beta = torch.nn.Parameter(self.beta, requires_grad=True)

        self.training_step_outputs = [[],[],[],[]]+[[] for _ in range(self.NLayers)]
        self.validation_step_outputs = [[],[],[],[]]+[[] for _ in range(self.NLayers)]

    
    def lossF(self, x_rec, x, a_rec, a=None, y=None):
        nmse_x = ((torch.norm(x_rec-x,dim=(2,3))/torch.norm(x,dim=(2,3)))**2).mean()
        err_a = (torch.abs(a_rec-a)/(torch.abs(a)+1e-10)).mean()
        return nmse_x + self.lam_loss * err_a
        
    def psi(self, lam):
        x = []
        for i,iae in enumerate(self.modelIAE_list):
            _,PhiE = iae.encode(iae.anchorpoints)
            B = []
            for r in range(iae.NLayers):
                B.append(torch.einsum('ik,kjl->ijl', lam[:,i], PhiE[iae.NLayers-r-1]))
            x.append(iae.decode(B))
        return torch.stack(x, 1)

    def forward(self, y, lam0=None, a0=None):
        
        if lam0 is None:
            lam0 = torch.ones((y.shape[0], self.nb_sources, self.num_ap), device=self.device)/self.num_ap
        lam = torch.clone(lam0)

        if a0 is None:
            a0 = torch.ones((y.shape[0], self.nb_sources,y.shape[2]), device=self.device)/self.nb_sources #torch.rand((y.shape[0], self.nb_sources,y.shape[2]), device=self.device) #
        a = torch.clone(a0)

        lam_layer = [] # register output of each layers
        a_layer = []
        for r in range(self.NLayers):
            #update of lam
            psi_lam = self.psi(lam)           
            if self.version in ['CNN','CNN_v2']:
                for j in range(self.nb_sources):
                    lam[:,j] = lam[:,j] + getattr(self,'W'+str(j+1)+'_l'+str(r+1))((y-torch.einsum('ijkl,ijl->ikl', psi_lam, a)).swapaxes(1,2)).squeeze(1)
            else: # MLP as default choice
                lam = lam + torch.einsum('ijl,ijml->ijm', a, torch.einsum('jmk,ikl->ijml', self.W[r], y-torch.einsum('ijkl,ijl->ikl', psi_lam, a)))

            #update of a 
            if not self.update_A == 'No-updating':
                x = self.psi(lam)
                if self.update_A == 'GD':
                    a = a + self.beta[r] * torch.einsum('ikl,ijkl->ijl', y-torch.einsum('ijkl,ijl->ikl', x, a), x)
                        
                elif self.update_A == 'LS': 
                    a = torch.einsum('ikl,ikjl->ijl', y, torch.pinverse(x.squeeze(-1)).unsqueeze(-1))

                a = a * (a>0)

            # back to LFBS when self.update_A == 'No-updating'           

            lam_layer.append(lam)
            a_layer.append(a)  
            
        return a_layer, lam_layer

    def predict(self, y):
        '''
        y: numpy array
        '''
        Y = torch.as_tensor(y).float().to(self.device)
        with torch.no_grad():
            a, lam = self.forward(Y)
        a = a[-1].cpu().detach().numpy()
        x = self.psi(lam[-1]).cpu().detach().numpy()
        return a, x

    def training_step(self, batch, batch_idx):
        y, a, x = batch
        a_pred_layers, lam_pred_layers = self.forward(y)
        a_pred = a_pred_layers[-1]
        x_pred = self.psi(lam_pred_layers[-1])
        
        loss_layers = torch.stack([self.lossF(self.psi(lam_pred_layers[i]), x, a_pred_layers[i], a=a) for i in np.arange(self.NLayers)])
        loss = torch.einsum('i,i->i', self.layers_weights, loss_layers).sum()

        for i in range(self.NLayers):
            self.training_step_outputs[i].append(loss_layers[i])
        self.training_step_outputs[self.NLayers].append(((torch.norm(x_pred-x,dim=(2,3))/torch.norm(x,dim=(2,3)))**2).mean())
        self.training_step_outputs[self.NLayers+1].append((torch.abs(a_pred-a)/(torch.abs(a)+1e-10)).mean())
        self.training_step_outputs[self.NLayers+2].append(torch.max(a_pred,dim=0).values.squeeze())
        self.training_step_outputs[self.NLayers+3].append(torch.mean(a_pred,dim=0).squeeze())

        if self.update_A=='GD':
            for i in range(self.NLayers):
                self.log("beta/layer%d" % (i+1), self.beta[i], on_step=True)

        return loss
    
    def on_train_epoch_end(self):
        self.log("loss/training_loss", torch.stack(self.training_step_outputs[self.NLayers-1]).mean())
        self.log("err_x/training", torch.stack(self.training_step_outputs[self.NLayers]).mean())
        self.log("err_a/training", torch.stack(self.training_step_outputs[self.NLayers+1]).mean())
        a_max = torch.stack(self.training_step_outputs[self.NLayers+2]).max(0).values
        a_mean = torch.stack(self.training_step_outputs[self.NLayers+3]).mean(0)
        if self.nb_sources == 1:
            self.log("amplitudes/a_max_val", a_max)
            self.log("amplitudes/a_mean_val", a_mean)
        else:
            for i in range(len(a_max)):
                self.log("amplitudes/a%d_max_train"%(i+1), a_max[i])
                self.log("amplitudes/a%d_mean_train"%(i+1), a_mean[i])
        for i in range(self.NLayers):   
            self.log("layer_loss/layer%d"%(i+1), torch.stack(self.training_step_outputs[i]).mean())
        for l in self.training_step_outputs:
            l.clear()

    
    def validation_step(self, batch, batch_idx):
        y, a, x = batch
        a_pred_layers, lam_pred_layers = self.forward(y)
        a_pred = a_pred_layers[-1]
        x_pred = self.psi(lam_pred_layers[-1])
        loss_layers = torch.stack([self.lossF(self.psi(lam_pred_layers[i]), x, a_pred_layers[i], a=a) for i in np.arange(self.NLayers)])
        loss = torch.einsum('i,i->i', self.layers_weights, loss_layers).sum()


        for i in range(self.NLayers):
            self.validation_step_outputs[i].append(loss_layers[i])

        self.validation_step_outputs[self.NLayers].append(((torch.norm(x_pred-x,dim=(2,3))/torch.norm(x,dim=(2,3)))**2).mean())
        self.validation_step_outputs[self.NLayers+1].append((torch.abs(a_pred-a)/(torch.abs(a)+1e-10)).mean())
        self.validation_step_outputs[self.NLayers+2].append(torch.max(a_pred,dim=0).values.squeeze())
        self.validation_step_outputs[self.NLayers+3].append(torch.mean(a_pred,dim=0).squeeze())
        return {"validation_loss": loss}

    def on_validation_epoch_end(self):
        self.log("loss/val_loss", torch.stack(self.validation_step_outputs[self.NLayers-1]).mean())
        self.log("err_x/validation", torch.stack(self.validation_step_outputs[self.NLayers]).mean())
        self.log("err_a/validation", torch.stack(self.validation_step_outputs[self.NLayers+1]).mean())
        a_max = torch.stack(self.validation_step_outputs[self.NLayers+2]).max(0).values
        a_mean = torch.stack(self.validation_step_outputs[self.NLayers+3]).mean(0)
        if self.nb_sources == 1:
            self.log("amplitudes/a_max_val", a_max)
            self.log("amplitudes/a_mean_val", a_mean)
        else:
            for i in range(len(a_max)):
                self.log("amplitudes/a%d_max_val"%(i+1), a_max[i])
                self.log("amplitudes/a%d_mean_val"%(i+1), a_mean[i])
        for i in range(self.NLayers):   
            self.log("layer_loss/layer%d"%(i+1), torch.stack(self.validation_step_outputs[i]).mean())
        for l in self.validation_step_outputs:
            l.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

# ###############################################################################################################################################
# #
# # Training
# #
# ###############################################################################################################################################
# def get_train_args(learning_rate=1e-3,batch_size=64,default_root_dir='./CKPT',max_epochs=5000,accumulate_grad_batches=4,auto_scale_batch_size=False,auto_lr_find=False,enable_checkpointing=True,profiler=None):

#     return {"learning_rate":learning_rate,"batch_size":batch_size,"default_root_dir":default_root_dir,"max_epochs":max_epochs,"accumulate_grad_batches":accumulate_grad_batches,"auto_scale_batch_size":auto_scale_batch_size,"auto_lr_find":auto_lr_find,"enable_checkpointing":enable_checkpointing,"profiler":profiler}

# ###############################################################################################################################################
# # Trainer
# ###############################################################################################################################################

# def training_lightning(arg_model=None,arg_train=None,from_model=None):
#     from pytorch_lightning.loggers import TensorBoardLogger
#     import pickle
#     from torch.utils.data import TensorDataset, DataLoader
#     import os

#     if torch.cuda.is_available():
#         device = 'cuda'
#         kwargs = {}
#         acc = "gpu"
#         Xpus_per_trial = 1
#     else:
#         device = 'cpu'
#         kwargs = {}
#         acc = 'cpu'
#         Xpus_per_trial = 1

#     print("device USED: ",device)

#     if device == 'cuda': # if GPU
#         torch.backends.cudnn.benchmark = True

#     if arg_train is None:
#         arg_train = get_train_args()

#     if arg_model is None:
#         print("Please provide arguments for the IAE model")

#     # Initialize the data loader
#     eps = arg_train['eps']
#     dataset = arg_train['dataset']
#     bs = arg_train['batch_size']
#     nepochs = arg_train['nepochs']

#     if eps==0.:
#         with open('data/'+dataset+'.npy', "rb") as f:
#             Y, A, X = pickle.load(f)
#     else:
#         with open('data/'+dataset+'_scaling'+str(eps)+'.npy', "rb") as f:
#             Y, A, X = pickle.load(f)

#     idx_train = np.arange(4000)
#     idx_val = np.arange(4000,5000)
#     Xtrain = X[idx_train]
#     Ytrain = Y[idx_train]
#     Atrain = A[idx_train]
#     Xval = X[idx_val]
#     Yval = Y[idx_val]
#     Aval = A[idx_val]

#     train_dataloader = DataLoader(TensorDataset(torch.from_numpy(Ytrain).float(), torch.from_numpy(Atrain).float(), torch.from_numpy(Xtrain).float()), batch_size=bs)
#     val_dataloader = DataLoader(TensorDataset(torch.from_numpy(Yval).float(), torch.from_numpy(Aval).float(), torch.from_numpy(Xval).float()), batch_size=bs)


#     n = arg_model['NLayers']
#     if dataset == 'noisy_Ba133':
#         l = ['Models/cnn_test']
#     elif dataset == 'noisy_Co57':
#         l = ['Models/cnn_test2']
#     elif dataset == 'noisy_Cs137':
#         l = ['Models/cnn_test3']
#     elif 'mixture' in dataset:
#         l = ['Models/cnn_test', 'Models/cnn_test2', 'Models/cnn_test3']

#     layers_weights = arg_train['layers_weights']
#     if layers_weights == 'last': # no control on the intermediate layers
#         layer_weights = torch.zeros(n)
#         layer_weights[-1]=1. 
#     elif layers_weights == 'exponential': # exponential weights for loss on all layers
#         layer_weights = 2**torch.arange(n) 
#     elif layers_weights == 'linear': # linear weights for loss on all layers
#         layer_weights = torch.arange(n)+1 
#     elif layers_weights == 'uniform': # uniform weights for loss on all layers
#         layer_weights = torch.ones(n) 
#     else:
#         raise ValueError("argument --loss must be one of last, exponential, linear and uniform")

#     layer_weights = layer_weights/layer_weights.sum() #normalize weights

#     lam_loss = arg_model['lam_loss']
#     update_A = arg_model['update_A']
#     version = arg_model['version']
#     arg_GLPALM = get_GLPALM_args(fname_list=l, NLayers=n, lam_loss=lam_loss, layers_weights=layer_weights, update_A=update_A, version=version, device=device)
#     model = GLPALM(arg_GLPALM).to(device)

#     if update_A == 'No-updating':
#         model_name = 'GLFBS'
#     else:
#         model_name = 'GLPALM_'+update_A
#     if not version is None:
#         model_name = model_name + '_' +version
#     model_name = model_name +'_L'+str(n)+dataset[dataset.find('_'):]

#     if eps != 0:
#         model_name = model_name + '_scaling' + str(eps)

#     if lam_loss != 0:
#         model_name = model_name + '_ERec'+ str(lam_loss)
#     if layers_weights != 'last':
#         model_name = model_name + '_WL_' + layers_weights
#     if os.path.isfile('./Models/model_'+ model_name+'.pth'): 
#         model = load_model('./Models/model_'+model_name,device=device)
#         model_name = model_name + '_retrained'
#     print('model loaded: ', os.path.isfile('./Models/model_'+ model_name+'.pth'))  

#     logger = TensorBoardLogger(save_dir='./CKPT', version=model_name, name="lightning_logs")
#     trainer = pl.Trainer(callbacks=[], default_root_dir='./CKPT', max_epochs=nepochs, logger=logger)
#     # print(trainer.__dict__)
#     trainer.fit(model, train_dataloader, val_dataloader)
#     save_name = './Models/model_'+ model_name
#     save_model(model, save_name)

#     return model
