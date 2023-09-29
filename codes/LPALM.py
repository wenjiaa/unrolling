import torch
import numpy as np
import IAE_CNN_TORCH_v2 as cnn
from utils import projection_simplex_torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import pytorch_lightning as pl
import time

############################################################
# Saving model from dict
############################################################

def save_model(model,fname='model_LPALM'+str(time.time())):

    params = {"arg_train":model.arg_train, "iae_model_fname_list": model.iae_fname_list, "NLayers": model.NLayers, "W_diag": model.W_diag, "W_shared": model.W_shared, "update_A": model.update_A, 'beta_shared': model.beta_shared, 'lam': model.lam}
    
    torch.save({"model":model.state_dict(),"params":params}, fname+".pth")


############################################################
# Loading model from dict
############################################################

def load_model(fname,device="cpu"):

    model_in = torch.load(fname+".pth", map_location=device)
    params = model_in["params"]
    params["device"] = device
    model_state = model_in["model"]
    model = LPALM(input_arg=params)
    model.load_state_dict(model_state)

    if device=='cuda':
        model = model.cuda()

    return model

############################################################
# LPALM args
############################################################

def get_LPALM_args(fname_list = None, NLayers=10, W_diag=True, W_shared=False, update_A='GD', beta_shared=False, lam=1., learning_rate=1e-3, device='cpu'):
    return {"iae_model_fname_list": fname_list, "NLayers": NLayers, "W_diag": W_diag, "W_shared": W_shared, 'update_A':update_A, 'beta_shared': beta_shared, 'lam': lam,  "learning_rate":learning_rate, 'device':device}

############################################################
# Main code
############################################################
  
class LPALM(pl.LightningModule):
    '''
    iae_fname_list: IAE models corresponding aux sources to unmix
    NLayers: number of layers
    W_diag: W is a vector when W_diag is True and a marix otherwise
    W_shared: All the layers share a same W if W_shared is True 
    update_A: projected gradient descent/ projected Least-squares/ no updating (=>LFBS)
    beta_shared: All the layers share a same W if beta_shared is True 
    lam: weight of A-related term in the loss
    '''
    def __init__(self, input_arg=None, arg_train=None):
        super(LPALM,self).__init__()
        torch.set_default_tensor_type(torch.FloatTensor)
        
        if input_arg is None:
            print("Run the get_arg first")
        
        self.iae_fname_list = input_arg["iae_model_fname_list"]
        self.nb_sources = len(self.iae_fname_list)
        self.modelIAE_list = [cnn.load_model(fname, device=input_arg['device']) for fname in self.iae_fname_list]
        for modelIAE in self.modelIAE_list:
            for param in modelIAE.parameters():
                param.requires_grad = False

        self.NLayers = input_arg["NLayers"]

        self.arg_train=arg_train
        self.lr=self.arg_train["learning_rate"]

        if "W_diag" in input_arg:
            self.W_diag = input_arg["W_diag"]
        else:
            self.W_diag = True

        self.W = torch.ones(self.nb_sources) * 0.9 if self.W_diag else torch.ones(self.nb_sources,self.nb_sources) * 0.9

        self.W_shared = input_arg['W_shared']
        if not self.W_shared:
            self.W = self.W.repeat(self.NLayers, 1) if self.W_diag else self.W.repeat(self.NLayers, 1, 1) 
        
        self.W = torch.nn.Parameter(self.W, requires_grad=True)
        
        self.update_A = input_arg['update_A']
        self.beta_shared = input_arg['beta_shared']

        if self.update_A == 'GD':
            self.beta = torch.tensor(0.9)
            if not self.beta_shared:
                self.beta = self.beta.repeat(self.NLayers, 1)
            self.beta = torch.nn.Parameter(self.beta, requires_grad=True)

        if 'lam' in input_arg:
            self.lam = input_arg['lam'] # weight of estimation error of a in the loss
        else:
            self.lam = 0.

        self.training_step_outputs = [[],[],[],[],[]]
        self.validation_step_outputs = [[],[],[],[],[]]

    
    def lossF(self, x_rec, x, a_rec, a=None):
        if self.lam!=0 and a==None: 
            print('Need ground-truch A for the computation of full-supervised loss !')
        nmse_x = ((torch.norm(x_rec-x,dim=(2,3))/torch.norm(x,dim=(2,3)))**2).mean()
        err_a = (torch.abs(a_rec-a)/torch.abs(a)).mean()
        return nmse_x + self.lam * err_a
        
    def forward(self, y, x0=None, a0=None):
        
        if x0 is None:
            x0 = 0.5 * torch.ones((y.shape[0], self.nb_sources, y.shape[1], y.shape[2]), device=self.device)
        x = torch.clone(x0)

        if a0 is None:
            a0 = (torch.ones(x.shape[0],x.shape[1],x.shape[3])/3.).to(self.device)
        a = torch.clone(a0)

        x_layer = [] # register output of each layers
        a_layer = []
        for r in range(self.NLayers):
            #update of x
            if self.W_shared:
                if self.W_diag:
                    W_eq = self.W
                else:
                    W_eq = torch.einsum('jm,iml->j', self.W, a)
            else:
                if self.W_diag:
                    W_eq = self.W[r]
                else:
                    W_eq = torch.einsum('jm,iml->j', self.W[r], a)
            x = x + torch.einsum('j,ikl->ijkl', W_eq, (y-torch.einsum('ijkl,ijl->ikl', x, a)))
            
            #projection onto manifolds with help of IAE models
            for i in np.arange(self.nb_sources):
                PhiX,PhiE = self.modelIAE_list[i].encode(x[:,i].clone())
                B,_ = self.modelIAE_list[i].interpolator(PhiX,PhiE)
                x[:,i] = self.modelIAE_list[i].decode(B)

            #update of a 
            if not self.update_A == 'No-updating':
                if self.update_A == 'GD':
                    if self.beta_shared:
                        a = a + self.beta * torch.einsum('ikl,ijkl->ijl', y-torch.einsum('ijkl,ijl->ikl', x, a), x)
                    else:
                        a = a + self.beta[r] * torch.einsum('ikl,ijkl->ijl', y-torch.einsum('ijkl,ijl->ikl', x, a), x)
                        
                elif self.update_A == 'LS': # a=(y@x.T)/(x@x.T)^-1
                    a = torch.einsum('ikl,ikjl->ijl', y, torch.pinverse(x.squeeze(-1)).unsqueeze(-1))

                # a = projection_simplex_torch(a.squeeze(-1), axis=1).unsqueeze(-1) # projection on simplex 
                a = a * (a>0) # projection positivity
            
            x_layer.append(x)
            a_layer.append(a)
                
            
        return a_layer, x_layer 

    def predict(self, y):
        '''
        y: numpy array
        '''
        Y = torch.as_tensor(y).float()
        a, x = self.forward(Y)
        a = a[-1].detach().numpy()
        x = x[-1].detach().numpy()
        return a, x

    def training_step(self, batch, batch_idx):
        y, a, x = batch
        a_pred, x_pred = self.forward(y)
        loss = self.lossF(x_pred[-1], x, a_pred[-1], a=a)

        self.training_step_outputs[0].append(loss)
        self.training_step_outputs[1].append(((torch.norm(x_pred[-1]-x,dim=(2,3))/torch.norm(x,dim=(2,3)))**2).mean())
        self.training_step_outputs[2].append((torch.abs(a_pred[-1]-a)/torch.abs(a)).mean())
        self.training_step_outputs[3].append(torch.max(a_pred[-1],dim=0).values.squeeze())
        self.training_step_outputs[4].append(torch.mean(a_pred[-1],dim=0).squeeze())

        if self.update_A=='GD':
            if self.beta_shared:
                self.log("beta", self.beta, on_step=True)
            else:
                for i in range(self.NLayers):
                    self.log("beta/layer%d" % (i+1), self.beta[i], on_step=True)

        return loss
    
    def on_train_epoch_end(self):
        self.log("loss/training_loss", torch.stack(self.training_step_outputs[0]).mean())
        self.log("err_x/training", torch.stack(self.training_step_outputs[1]).mean())
        self.log("err_a/training", torch.stack(self.training_step_outputs[2]).mean())
        a_max = torch.stack(self.training_step_outputs[3]).max(0).values
        a_mean = torch.stack(self.training_step_outputs[4]).mean(0)
        if self.nb_sources == 1:
            self.log("amplitudes/a_max_val", a_max)
            self.log("amplitudes/a_mean_val", a_mean)
        else:
            for i in range(len(a_max)):
                self.log("amplitudes/a%d_max_train"%(i+1), a_max[i])
                self.log("amplitudes/a%d_mean_train"%(i+1), a_mean[i])
        for l in self.training_step_outputs:
            l.clear()

    
    def validation_step(self, batch, batch_idx):
        y, a, x = batch
        a_pred, x_pred = self.forward(y)
        loss = self.lossF(x_pred[-1], x, a_pred[-1], a=a)
        
        self.validation_step_outputs[0].append(loss)
        self.validation_step_outputs[1].append(((torch.norm(x_pred[-1]-x,dim=(2,3))/torch.norm(x,dim=(2,3)))**2).mean())
        self.validation_step_outputs[2].append((torch.abs(a_pred[-1]-a)/torch.abs(a)).mean())
        self.validation_step_outputs[3].append(torch.max(a_pred[-1],dim=0).values.squeeze())
        self.validation_step_outputs[4].append(torch.mean(a_pred[-1],dim=0).squeeze())
        return {"validation_loss": loss}

    def on_validation_epoch_end(self):
        self.log("loss/val_loss", torch.stack(self.validation_step_outputs[0]).mean())
        self.log("err_x/validation", torch.stack(self.validation_step_outputs[1]).mean())
        self.log("err_a/validation", torch.stack(self.validation_step_outputs[2]).mean())
        a_max = torch.stack(self.validation_step_outputs[3]).max(0).values
        a_mean = torch.stack(self.validation_step_outputs[4]).mean(0)
        if self.nb_sources == 1:
            self.log("amplitudes/a_max_val", a_max)
            self.log("amplitudes/a_mean_val", a_mean)
        else:
            for i in range(len(a_max)):
                self.log("amplitudes/a%d_max_val"%(i+1), a_max[i])
                self.log("amplitudes/a%d_mean_val"%(i+1), a_mean[i])
        for l in self.validation_step_outputs:
            l.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    