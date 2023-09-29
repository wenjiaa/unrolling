"""
Metric Learning

To be done :
    - Log barrier for positivity
    - Lightning
    - New BSP
"""

import pickle
import numpy as np
import time
import sys
import os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam,NAdam,AdamW, SGD, Adagrad, LBFGS,ASGD
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
from pytorch_lightning.callbacks import StochasticWeightAveraging,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

############################################################
# Main code
############################################################

def _get_optimizer(Optimizer,parameters,learning_rate=1e-4):

    if Optimizer == 0:
        print("Adam")
        optimizer = Adam(parameters, lr=learning_rate)
    elif Optimizer == 1:
        print("AdamW")
        optimizer = AdamW(parameters, lr=learning_rate)
    elif Optimizer == 2:
        print("NAdam")
        optimizer = NAdam(parameters, lr=learning_rate)
    elif Optimizer == 3:
        print("Adagrad")
        optimizer = Adagrad(parameters, lr=learning_rate, weight_decay=1e-5)
    elif Optimizer == 4:
        print("SGD")
        optimizer = SGD(parameters, lr=learning_rate)
    elif Optimizer == 5:
        print("ASGD")
        optimizer = ASGD(parameters, lr=learning_rate)

    return optimizer

def _normalize(X,norm='1',log=False,torchtensor=False):

    if torchtensor:
        if len(X.shape) < 3:
            if log:
                X = np.log10(X)
            if norm == '1':
                A = torch.sum(torch.abs(X),(1))
            if norm == '2':
                A = torch.sqrt(torch.sum(torch.square(X),(1)))
            if norm == 'inf':
                A = torch.max(2*torch.abs(X),dim=1).values
            return np.einsum('ij,i-> ij',X,1/A),A
        else:
            if log:
                Y = np.log10(X)
            if norm == '1':
                A = torch.sum(torch.abs(X),(1,2))
            if norm == '2':
                A = torch.sqrt(torch.sum(torch.square(X),(1,2)))
            if norm == 'inf':
                A = torch.max(torch.max(2*torch.abs(X),dim= 1),dim=2) # Not quite clean
            return np.einsum('ijk,i-> ijk',X,1/A),A
    else:
        if len(X.shape) < 3:
            if log:
                X = np.log10(X)
            if norm == '1':
                A = np.sum(abs(X),axis=1)
            if norm == '2':
                A = np.sqrt(np.sum(X**2,axis=1))
            if norm == 'inf':
                A = 2*np.max(X,axis=1)
            return np.einsum('ij,i-> ij',X,1/A),A
        else:
            if log:
                Y = np.log10(X)
            if norm == '1':
                A = np.sum(abs(X),axis=(1,2))
            if norm == '2':
                A = np.sqrt(np.sum(X**2,axis=(1,2)))
            if norm == 'inf':
                A = 2*abs(X).max(axis=(1,2))
            return np.einsum('ijk,i-> ijk',X,1/A),A

def _loss(LossOpt='l2'):

    """
    Defines loss functions
    """

    if LossOpt=='l1':
        return torch.nn.L1Loss()
    elif LossOpt=='kl':
        return torch.nn.KLDivLoss()
    else:
        return torch.nn.MSELoss()

    ###
    #  ADDING CORRUPTION
    ###

def _corrupt(x,noise_level=None,GaussNoise=True,device="cpu"): # Corrupting the data // THIS COULD BE CHANGED // should be tested

    if GaussNoise:
        noise = noise_level*torch.randn_like(x).to(device)
        return x + noise
    else:
        noise = torch.bernoulli(noise_level*torch.ones_like(x.data)).to(device)
        return x * noise # We could put additive noise

############################################################
# Saving model from dict
############################################################

def save_model(model,fname='test'):

    params = {"arg_train":model.arg_train,"mean_lambda":model.mean_lambda,"version":model.version,"reg_inv":model.reg_inv,"normalisation":model.normalisation,"anchorpoints":model.anchorpoints, "nsize_fsize":model.nsize_fsize, "nsize_fstride":model.nsize_fstride, "nsize_fnum":model.nsize_fnum,"rho_latcon":model.rho_latcon,"simplex" : model.simplex,"device":model.device,"nonneg_weights":model.nonneg_weights}

    torch.save({"model":model.state_dict(),"iae_params":params}, fname+".pth")

############################################################
# Loading model from dict
############################################################

def load_model(fname,device="cpu"):

    model_in = torch.load(fname+".pth", map_location=device)
    params = model_in["iae_params"]
    model_state = model_in["model"]

    iae = IAE(input_arg=params,model_load=True)
    iae.load_state_dict(model_state)

    if device=='cuda':
        iae = iae.cuda()

    return iae

############################################################
# IAE args
############################################################

def get_IAE_args(mean_lambda=False,normalisation='inf',anchorpoints=None, nsize_fstride=None, nsize_fnum=None, nsize_fsize=None,
             simplex=False,nonneg_weights=False, rho_latcon=None,reg_inv=1e-6,device="cpu",dropout_rate=None,version="version_December_2022"):
    return {'mean_lambda':mean_lambda,'rho_latcon':rho_latcon,'normalisation':normalisation,'anchorpoints':anchorpoints, 'nsize_fnum':nsize_fnum,'nsize_fstride':nsize_fstride,'nsize_fsize':nsize_fsize, 'reg_inv':reg_inv,'simplex':simplex, 'nonneg_weights':nonneg_weights,'device':device,'dropout_rate':dropout_rate,'version':version}

############################################################
# Main code
############################################################

class IAE(pl.LightningModule):
    """
    Model - input IAE model, overrides other parameters if provided (except the number of layers)
    fname - filename for the IAE model
    anchorpoints - anchor points
    nsize - network structure (e.g. [8, 8, 8, 8] for a 3-layer neural network of size 8)
    active_forward - activation function in the encoder
    active_backward - activation function in the decoder
    res_factor - residual injection factor in the ResNet-like architecture
    reg_parameter - weighting constant to balance between the sample and transformed domains
    cost_weight - weighting constant to balance between the sample and transformed domains in the learning stage
    reg_inv - regularization term in the barycenter computation
    simplex - simplex constraint onto the barycentric coefficients
    nneg_weights - non-negative constraint onto the barycentric coefficients
    nneg_output - non-negative constraint onto the output
    noise_level - noise level in the learning stage as in the denoising autoencoder
    cost_type - cost function (not used)
    optim_learn - optimization algorithm in the learning stage
        (0: Adam, 1: Momentum, 2: RMSprop, 3: AdaGrad, 4: Nesterov, 5: SGD)
    optim_proj - optimization algorithm in the barycentric span projection
    step_size - step size of the optimization algorithms
    niter - number of iterations of the optimization algorithms
    eps_cvg - convergence tolerance
    verb - verbose mode
    """

    def __init__(self, input_arg=None,arg_train=None,config=None,model_load=False):
        """
        Initialization
        """
        super(IAE,self).__init__()

        if input_arg is None:
            print("Run the get_arg first")


        self.anchorpoints = torch.as_tensor(input_arg["anchorpoints"])
        self.simplex = input_arg["simplex"]
        self.num_ap = input_arg["anchorpoints"].shape[0]
        #self.device = input_arg["device"]
        self.nonneg_weights = input_arg["nonneg_weights"]
        self.normalisation = input_arg["normalisation"]
        self.version = input_arg["version"]
        self.reg_inv=input_arg["reg_inv"]
        self.normalisation=input_arg["normalisation"]
        self.Lin = self.anchorpoints.shape[1]
        self.PhiE = None
        #self.device=input_arg["device"]
        self.mean_lambda=input_arg["mean_lambda"]

        if model_load:
            self.arg_train=input_arg["arg_train"]
        else:
            self.arg_train=arg_train
        self.lr=self.arg_train["learning_rate"]

        self.LossF = _loss(self.arg_train["LossOpt"])

        if input_arg["rho_latcon"] is None:
            self.rho_latcon = torch.ones((self.NLayers,),device=self.device)
        else:
            self.rho_latcon = input_arg["rho_latcon"]

        ##################### OPTIONS FOR RAY TUNE

        self.nsize_fsize = input_arg["nsize_fsize"]
        self.nsize_fnum = input_arg["nsize_fnum"]
        self.nsize_fstride = input_arg["nsize_fstride"]

        if config is not None:
            if "lr" in config:
                self.lr=config["lr"]
            if "fsizefactor" in config:
                self.nsize_fsize = config["fsizefactor"]*input_arg["nsize_fsize"]
            if "nfilterfactor" in config:
                self.nsize_fnum= config["nfilterfactor"]*input_arg["nsize_fnum"]
            if "rholatconfactor" in config:
                self.rho_latcon = config["rholatconfactor"]*input_arg["rho_latcon"]
        self.NLayers = len(self.nsize_fsize)

        #####################

        dim = []
        dim.append(self.Lin)
        Lin = self.Lin

        for r in range(self.NLayers):

            if r ==0:
                Nch_in = self.anchorpoints.shape[2]
            else:
                Nch_in = self.nsize_fnum[r-1]
            Nch_out = self.nsize_fnum[r]
            kern_size = self.nsize_fsize[r]
            stride = self.nsize_fstride[r]
            Lout = np.int(np.floor(1+1/stride*(Lin-kern_size)))
            dim.append(Lout)
            Lin = Lout

            encoder = []
            encoder.append(torch.nn.Conv1d(Nch_in, Nch_out,kern_size,stride=stride,bias=False))
            encoder.append(torch.nn.BatchNorm1d(Nch_out))
            encoder.append(torch.nn.ELU())  # This could be changed based

            setattr(self,'encoder'+str(r+1),torch.nn.Sequential(*encoder))

            # For the lateral connection

            if r >0 and r < self.NLayers:

                encoder = []
                encoder.append(torch.nn.Conv1d(Nch_in, Nch_out,kern_size,stride=stride,bias=False))
                encoder.append(torch.nn.BatchNorm1d(Nch_out))
                encoder.append(torch.nn.ELU())  # This could be changed based

                setattr(self,'encoder_lat'+str(r),torch.nn.Sequential(*encoder))

        self.dim = dim

        for r in range(1,self.NLayers+1):
            if r == self.NLayers:
                Nch_out = self.anchorpoints.shape[2]
            else:
                Nch_out = self.nsize_fnum[self.NLayers-r-1]
            Nch_in = self.nsize_fnum[self.NLayers-r]
            kern_size = self.nsize_fsize[self.NLayers-r]
            stride = self.nsize_fstride[self.NLayers-r]

            decoder = []
            decoder.append(torch.nn.ConvTranspose1d(Nch_in, Nch_out,kern_size,bias=False))

            if r < self.NLayers+1:
                decoder.append(torch.nn.ELU())  # This could be changed based

            setattr(self,'decoder'+str(r),torch.nn.Sequential(*decoder))

            # For the lateral connection
            if r < self.NLayers:

                decoder = []
                decoder.append(torch.nn.ConvTranspose1d(Nch_in, Nch_out,kern_size,bias=False))
                decoder.append(torch.nn.ELU())  # This could be changed based

                setattr(self,'decoder_lat'+str(r),torch.nn.Sequential(*decoder))

    ###
    #  DISPLAY
    ###

    def display(self,epoch,epoch_time,train_acc,rel_acc,pref="Learning stage - ",niter=None):

        if niter is None:
            niter = self.niter

        percent_time = epoch/(1e-12+niter)
        n_bar = 50
        bar = ' |'
        bar = bar + '█' * int(n_bar * percent_time)
        bar = bar + '-' * int(n_bar * (1-percent_time))
        bar = bar + ' |'
        bar = bar + np.str(int(100 * percent_time))+'%'
        m, s = divmod(np.int(epoch*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run = ' [{:d}:{:02d}:{:02d}<'.format(h, m, s)
        m, s = divmod(np.int((niter-epoch)*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run += '{:d}:{:02d}:{:02d}]'.format(h, m, s)

        sys.stdout.write('\033[2K\033[1G')
        if epoch_time > 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- validation loss = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
        if epoch_time < 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- validation loss = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4}'.format(1./epoch_time)+' epoch/s', end="\r")
    ###
    #  ENCODE
    ###

    def encode(self, X):

        PhiX_lat = []
        PhiE_lat = []

        PhiX = getattr(self,'encoder'+str(1))(torch.swapaxes(X,1,2))
        PhiE = getattr(self,'encoder'+str(1))(torch.swapaxes(self.anchorpoints.clone(),1,2))

        for r in range(1,self.NLayers):

            if r < self.NLayers:
                PhiX_lat.append(torch.swapaxes(getattr(self,'encoder_lat'+str(r))(PhiX),1,2))
                PhiE_lat.append(torch.swapaxes(getattr(self,'encoder_lat'+str(r))(PhiE),1,2))

            PhiX = getattr(self,'encoder'+str(r+1))(PhiX)
            PhiE = getattr(self,'encoder'+str(r+1))(PhiE)

        PhiX_lat.append(torch.swapaxes(PhiX,1,2))
        PhiE_lat.append(torch.swapaxes(PhiE,1,2))

        return PhiX_lat,PhiE_lat

    def decode(self, B):

        Xrec = torch.swapaxes(B[0],1,2)

        for r in range(self.NLayers-1):

            Xtemp = Xrec
            Btemp = torch.swapaxes(B[r+1],1,2)

            up = torch.nn.Upsample(size=self.dim[self.NLayers-r-1], mode='linear', align_corners=True)
            Xrec = up(getattr(self,'decoder'+str(r+1))(Xtemp) + self.rho_latcon[r]*getattr(self,'decoder_lat'+str(r+1))(Btemp))

        Xrec = getattr(self,'decoder'+str(self.NLayers))(Xrec)
        up = torch.nn.Upsample(size=self.dim[0], mode='linear', align_corners=True)
        Xrec = up(Xrec)

        return torch.swapaxes(Xrec,1,2)

    def interpolator(self, PhiX, PhiE):

        L = []
        B = []

        if self.mean_lambda:

            r=0
            PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
            iPhiE = torch.linalg.inv(PhiE2 + self.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
            Lambda = torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)
            sum_val = 1

            for r in range(1,self.NLayers):
                PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
                iPhiE = torch.linalg.inv(PhiE2 + self.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
                Lambda += self.rho_latcon[r-1]*torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)
                sum_val += self.rho_latcon[r-1]
            Lambda=Lambda/sum_val

            if self.nonneg_weights:
                mu = torch.max(Lambda,dim=1).values-1.0
                for i in range(2*Lambda.shape[1]) : # or maybe more
                    F = torch.sum(torch.maximum(Lambda,mu.reshape(-1,1)), dim=1) - Lambda.shape[1]*mu-1
                    mu = mu + 1/Lambda.shape[1]*F
                Lambda = torch.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

            elif self.simplex:
                ones = torch.ones_like(Lambda,device=self.device)
                mu = (1 - torch.sum(Lambda,dim=1))/torch.sum(iPhiE)
                Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)

            for r in range(self.NLayers):
                L.append(Lambda)
                B.append(torch.einsum('ik,kjl->ijl', Lambda, PhiE[self.NLayers-r-1]))

        else:

            for r in range(self.NLayers):
                PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
                iPhiE = torch.linalg.inv(PhiE2 + self.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
                Lambda = torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)

                if self.nonneg_weights:
                    mu = torch.max(Lambda,dim=1).values-1.0
                    for i in range(2*Lambda.shape[1]) : # or maybe more
                        F = torch.sum(torch.maximum(Lambda,mu.reshape(-1,1)), dim=1) - Lambda.shape[1]*mu-1
                        mu = mu + 1/Lambda.shape[1]*F
                    Lambda = torch.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

                elif self.simplex:
                    ones = torch.ones_like(Lambda,device=self.device)
                    mu = (1 - torch.sum(Lambda,dim=1))/torch.sum(iPhiE)
                    Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)

                L.append(Lambda)
                B.append(torch.einsum('ik,kjl->ijl', Lambda, PhiE[self.NLayers-r-1]))

        return B, L

    def fast_interpolation(self, X, Amplitude=None):

        # Estimating the amplitude

        if Amplitude is None:
            _,Amplitude = _normalize(X,norm=self.normalisation)

        # Encode data
        X = torch.as_tensor(X.astype('float32'))
        PhiX,PhiE = self.encode(torch.einsum('ijk,i->ijk',X,torch.as_tensor(1./Amplitude.astype('float32'))))

        # Define the barycenter
        B, Lambda = self.interpolator(PhiX,PhiE)

        # Decode the barycenter
        XRec = self.decode(B)

        if X.shape[0]==1:
            XRec = torch.einsum('ijk,i->ijk',XRec,torch.as_tensor(Amplitude))
        else:
            XRec = torch.einsum('ijk,i->ijk',XRec,torch.as_tensor(Amplitude.squeeze()))

        Output = {"PhiX": PhiX, "PhiE": PhiE, "Barycenter": B, "Lambda": Lambda, "Amplitude": Amplitude, "XRec": XRec}

        return Output

    def get_barycenter(self, Lambda, Amplitude=None):

        _,PhiE = self.encode(self.anchorpoints)

        if Amplitude is None:
            # print("To be done")
            Amplitude = torch.ones(Lambda[0].shape[0]).to(self.device)

        B = []
        for r in range(self.NLayers):
            B.append(torch.einsum('ik,kjl->ijl', Lambda[r], PhiE[self.NLayers-r-1]))

        # Decode the barycenter
        XRec = torch.einsum('ijk,i -> ijk',self.decode(B),Amplitude)

        return XRec

    def forward(self, x):
        Z,Ze = self.encode(x)
        B,_ = self.interpolator(Z,Ze)
        return self.decode(B)

    def training_step(self, batch, batch_idx):

        # Corrupting the data

        if self.arg_train["noise_level"] is not None:
            x = _corrupt(batch,self.arg_train["noise_level"],self.arg_train["GaussNoise"],device=self.device)
        else:
            x = batch

        # Applying the IAE

        Z,Ze = self.encode(x)
        B,_ = self.interpolator(Z,Ze)
        x_hat = self.decode(B)

        # Computing the cost

        cost = 0
        for r in range(self.NLayers):
            cost += self.rho_latcon[r]* self.LossF(Z[self.NLayers-r-1],B[r])

        loss = (1+self.arg_train["reg_parameter"])*(self.LossF(x_hat, x) + self.arg_train["reg_parameter"]*cost)

        if self.arg_train["nonneg_output"]:
            loss+= -torch.mean(torch.log(1e-16 + x_hat))

        self.log("train_loss", loss, on_step=True)
        self.log("reg_train_loss", cost, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        Z,Ze = self.encode(x)
        B,_ = self.interpolator(Z,Ze)
        x_hat = self.decode(B)
        cost = self.LossF(Z[self.NLayers-1],B[0])
        for r in range(1,self.NLayers):
            cost += self.rho_latcon[r-1]* self.LossF(Z[self.NLayers-r-1],B[r])
        acc = -20*torch.log10(self.LossF(x_hat, x)+1e-16)
        loss = (1+self.arg_train["reg_parameter"])*(acc + self.arg_train["reg_parameter"]*cost)

        if self.arg_train["nonneg_output"]:
            loss+= -torch.mean(torch.log(1e-16 + x_hat))

        self.log("validation_loss", loss)
        self.log("reg_validation_loss", cost)
        self.log("validation_accuracy", acc)

        return {"validation_loss": loss, "validation_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["validation_accuracy"] for x in outputs]).mean()
        #print(outputs)
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def configure_optimizers(self):
        optimizer = _get_optimizer(self.arg_train["Optimizer"],self.parameters(),learning_rate=self.lr)
        return optimizer


###############################################################################################################################################
#
# Training
#
###############################################################################################################################################

def get_train_args(fname='IAE_model',SWA_niter=None,nonneg_output=False,verb=True,GaussNoise=True,noise_level=None,reg_parameter=0.1,learning_rate=1e-3,batch_size=64,Optimizer=0,normalisation='1',LossOpt='l2',default_root_dir='./CKPT',max_epochs=5000,accumulate_grad_batches=4,auto_scale_batch_size=False,auto_lr_find=False,enable_checkpointing=True,profiler=None):

    return {"SWA_niter":SWA_niter,"fname":fname,"nonneg_output":nonneg_output,"verb":verb,"GaussNoise":GaussNoise,"noise_level":noise_level,"reg_parameter":reg_parameter,"learning_rate":learning_rate,"batch_size":batch_size,"Optimizer":Optimizer,"normalisation":normalisation,"LossOpt":LossOpt,"default_root_dir":default_root_dir,"max_epochs":max_epochs,"accumulate_grad_batches":accumulate_grad_batches,"auto_scale_batch_size":auto_scale_batch_size,"auto_lr_find":auto_lr_find,"enable_checkpointing":enable_checkpointing,"profiler":profiler}

###############################################################################################################################################
# Trainer
###############################################################################################################################################

def training_lightning(XTrain,arg_IAE=None,arg_train=None,from_model=None,Xvalidation=None,checkmodel=False,checkbest=False):

    """
    CPUData : if true, keeps the data local and only transfer the batches to the GPU
    """

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

    if arg_train is None:
        arg_train = get_train_args()

    if arg_IAE is None:
        print("Please provide arguments for the IAE model")

    ###
    ### normalisation
    ###

    XTrain = torch.as_tensor(_normalize(XTrain,norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
    arg_IAE["anchorpoints"] = torch.as_tensor(_normalize(arg_IAE["anchorpoints"],norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
    data_loader = DataLoader(XTrain, batch_size=arg_train["batch_size"], shuffle=True, **kwargs)

    # Initialize the data loader

    if Xvalidation is not None:
        Xvalidation = torch.as_tensor(_normalize(Xvalidation,norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
        validation_loader = DataLoader(Xvalidation, batch_size=arg_train["batch_size"], shuffle=False, **kwargs)
    else:
        validation_loader = None

    ###
    ###
    ###

    if from_model is not None:
        IAEmodel = load_model(from_model,device=device)
        # This should be done more systematically
        IAEmodel.nonneg_weights = arg_IAE["nonneg_weights"]
        print(IAEmodel.nonneg_weights)
    else:
        IAEmodel = IAE(input_arg=arg_IAE,arg_train=arg_train)
    IAEmodel = IAEmodel.to(device)

    if arg_train["verb"]:
        print("Training step")

    mycb = [] #[StochasticWeightAveraging(swa_lrs=1e-3,swa_epoch_start=2500)]

    trainer = pl.Trainer(callbacks=mycb,default_root_dir=arg_train["default_root_dir"],max_epochs=arg_train["max_epochs"],accumulate_grad_batches=arg_train["accumulate_grad_batches"],auto_scale_batch_size=arg_train["auto_scale_batch_size"],auto_lr_find=arg_train["auto_lr_find"],enable_checkpointing=arg_train["enable_checkpointing"],profiler=arg_train["profiler"],accelerator=acc, devices=Xpus_per_trial)
    out_train = trainer.fit(IAEmodel, data_loader,) # Callback pour la validation

    if arg_train["SWA_niter"] is not None:

        mycb = [StochasticWeightAveraging(swa_lrs=1e-3,swa_epoch_start=1)]
        trainer = pl.Trainer(callbacks=mycb,default_root_dir=arg_train["default_root_dir"],max_epochs=arg_train["SWA_niter"],accumulate_grad_batches=arg_train["accumulate_grad_batches"],auto_scale_batch_size=arg_train["auto_scale_batch_size"],auto_lr_find=arg_train["auto_lr_find"],enable_checkpointing=arg_train["enable_checkpointing"],profiler=arg_train["profiler"],accelerator=acc, devices=Xpus_per_trial)
    out_train = trainer.fit(IAEmodel, data_loader,) # Callback pour la validation

    if checkbest:
        if arg_train["verb"]:
            print("Validation step")
        out_val = trainer.validate(IAEmodel, validation_loader,ckpt_path="best",verbose=True)
    else:
        out_val = None

    # Saving the model

    if from_model is not None:
        fname_out = from_model+'_restart'
    else:
        fname_out = arg_train["fname"]
    save_model(IAEmodel,fname=fname_out)

    return IAEmodel, out_train, out_val

###############################################################################################################################################
# Parameter fitting
###############################################################################################################################################

def P_circlel1_nneg(u,radius):
    f0 = np.concatenate(([0],np.sort(u[u>0])))
    if np.sum(f0) > radius:
        ccf = np.cumsum(f0[::-1])[::-1] - f0*np.linspace(len(f0),1,len(f0))
        i = np.max(np.where(ccf > radius))
        t = (ccf[i+1]-radius)/( ccf[i+1]-ccf[i] ) * (f0[i]-f0[i+1]) + f0[i+1]
        ut = (u - t)*(u > t)
    else:
        ut = u*(u > 0)
        t = 1/len(ut[ut > 0])*(radius - np.sum(ut))
        ut[ut > 0] = ut[ut > 0] + t
    return ut

def P_circlel1(u,radius):
    f0 = np.concatenate(([0],np.sort(np.abs(u))))
    if np.sum(f0) > radius:
        ccf = np.cumsum(f0[::-1])[::-1] - f0*np.linspace(len(f0),1,len(f0))
        i = np.max(np.where(ccf > radius))
        t = (ccf[i+1]-radius)/( ccf[i+1]-ccf[i] ) * (f0[i]-f0[i+1]) + f0[i+1]
        ut = (u - t*np.sign(u))*(abs(u) > t)
    else:
        ut = u*(u > 0)
        t = 1/len(u)*(radius - np.sum(np.abs(u)))
        ut = u + t*np.sign(u)
    return ut

def QuickSampling(x,model,npoints=100,simplex=True,nonneg_weights=False,Lrange=1,verb=False):
    """
    Quick sampling to initialise the BSP
    """

    _,Amplitude = _normalize(x,norm=model.normalisation)
    Lambda0 = 2*Lrange*(np.random.rand(npoints,model.anchorpoints.shape[0])-0.5)
    if simplex:
        Lambda = np.array([P_circlel1(Lambda0[r],1) for r in range(npoints)])
    elif nonneg_weights:
        Lambda = np.array([P_circlel1_nneg(Lambda0[r],1) for r in range(npoints)])
    else:
        Lambda = Lambda0
    rec = _get_barycenter(Lambda,model=model)
    err = -20*np.log10(np.linalg.norm(rec-x/Amplitude,axis=(1,2))/np.linalg.norm(x/Amplitude))
    I = np.where(err == np.max(err))
    if verb:
        print("Lambda = ",Lambda[I])
        print("Error in dB = ",err[I])
    return Lambda[I],rec[I]*Amplitude

def bsp(x,model=None,fname=None,a0=None,Lambda0=None,epochs= 1000,LossOpt='l2',line_search_fn='strong_wolfe',tol=1e-6):
    """
    This is just an example code to see how the IAE model could be used as a generative model
    WITH BFGS - Clean it up + amplitude
    """

    from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)

    PhiE,_ = model.encode(model.anchorpoints)
    d = model.anchorpoints.shape[0]
    b,tx,ty = x.shape

    # Initialize Lambda0

    loss_val = []

    if a0 is None:
        _,a = _normalize(x,norm=model.normalisation)
    else:
        a = a0

    if Lambda0 is None:
        Lambda = model.fast_interpolation(x,Amplitude=a0)["Lambda"][0].detach().numpy()
    else:
        Lambda = Lambda0

    F = _loss(LossOpt=LossOpt)

    x = torch.as_tensor(x.astype("float32"))

    ####

    def Func(P):
        P = P.reshape(b,-1)
        B = []
        for r in range(model.NLayers):
            B.append(torch.einsum('ik,kjl->ijl',P[:,1::] , PhiE[model.NLayers-r-1]))
        return torch.einsum('ijk,i -> ijk',model.decode(B),P[:,0])

    P = np.concatenate((a.reshape(-1,1),Lambda),1).reshape(-1,)
    Params = torch.tensor(P.astype('float32'), requires_grad=True)
    optimizer = torch.optim.LBFGS([Params],max_iter=epochs,line_search_fn=line_search_fn,tolerance_change=tol)

    def Loss():
        optimizer.zero_grad()
        rec = Func(Params)
        loss = F(rec, x)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(Loss)

    Params = Params.reshape(b,-1)
    a,L = Params[:,0].detach().numpy().squeeze(),Params[:,1::]

    return _get_barycenter(L.detach().numpy(),amplitude=a,model=model),L.detach().numpy(),a


def _get_barycenter(Lambda,amplitude=None,model=None,fname=None):

    """
    Reconstruct a barycenter from Lambda
    """
    #from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)

    PhiE,_ = model.encode(model.anchorpoints)

    B = []
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',torch.as_tensor(Lambda.astype("float32")), PhiE[model.NLayers-r-1]))

    if  amplitude is None:
        return model.decode(B).detach().numpy()
    else:
        return torch.einsum('ijk,i -> ijk',model.decode(B),torch.as_tensor(amplitude.astype("float32"))).detach().numpy()

###############################################################################################################################################
##############################################################################################################################################
# Check models
###############################################################################################################################################

def CheckModels(data,Models,Names=None,display=False,SNRVal=None):

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex']=False
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams.update({'font.size': 22})
    from matplotlib.ticker import MaxNLocator
    vcol = ['mediumseagreen','crimson','steelblue','darkmagenta','burlywood','khaki','lightblue','darkseagreen','deepskyblue','forestgreen','gold','indianred','midnightblue','olive','orangered','orchid','red','steelblue']

    def nmse(x1,x2):
        return -20*np.log10(np.linalg.norm(x1-x2)/np.linalg.norm(x2)),-20*np.log10(np.linalg.norm(x1-x2,axis=(1,2))/np.linalg.norm(x2,axis=(1,2)))

    if SNRVal is None:
        SNRVal = np.array([-10,-5,0,5,10,15,20,25,30,35,40]) # Should depend on the normalise as well

    all_nmse = []
    nmse_ind = []
    all_nmse_ind = []
    norm_data = np.linalg.norm(data)

    for mod in Models:
        nmseloc = []
        nmseloc_ind = []
        model = load_model(mod)
        for r in SNRVal:
            noise = np.random.randn(*data.shape)
            Xn = data + 10**(-r/20)*norm_data*noise/np.linalg.norm(noise)
            rec = model.fast_interpolation(Xn)
            xrec = rec["XRec"].data
            nmse_tot,nmse_ind = nmse(xrec,data)
            nmseloc_ind.append(np.array(nmse_ind))
            nmseloc.append(nmse_tot)
        all_nmse_ind.append(nmseloc_ind)
        all_nmse.append(np.array(nmseloc))

    if display:
        plt.figure(figsize=(15,10))
        for r in range(len(Models)):
            if Names is not None:
                name = Names[r]
            else:
                name = Models[r]
            plt.plot(SNRVal,all_nmse[r],color=vcol[r], marker='o', linestyle='dashed',linewidth=3, markersize=12,label=name)
        plt.legend()
        plt.xlabel('noise level in dB')
        plt.ylabel('NMSE in dB')

        for q in range(len(SNRVal)):
            plt.figure(figsize=(15,10))
            if Names is not None:
                name = Names[r]
            else:
                name = Models[r]
            for r in range(len(Models)):
                plt.hist(all_nmse_ind[r][q],100,color=vcol[r],alpha=0.3,label=name+' '+str(SNRVal[q])+'dB')
            plt.legend()
            plt.xlabel('NMSE in dB')

    # Add something with the latent space parameters

    Lrec = rec["Lambda"][0].data # For the last layer
    plt.figure(figsize=(15,10))
    plt.plot(Lrec[:,0], Lrec[:,1],'o')
    plt.title("Representation in the latent space - 2D vis")

    if Lrec.shape[1] > 2:
        plt.figure(figsize=(15,10))
        ax = plt.axes(projection ='3d')
        ax.plot3D(Lrec[:,0], Lrec[:,1], Lrec[:,2],'o',color= 'green')
        plt.title("Representation in the latent space")

    return all_nmse

def CheckOptimisationLandscape(data,Models,Names=None,display=False,Lambda0=None,SNRVal=40,npoints=32,Lrange=1,elev=45,azim=45):

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex']=False
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams.update({'font.size': 22})
    from matplotlib.ticker import MaxNLocator
    vcol = ['mediumseagreen','crimson','steelblue','darkmagenta','burlywood','khaki','lightblue','darkseagreen','deepskyblue','forestgreen','gold','indianred','midnightblue','olive','orangered','orchid','red','steelblue']

    def nmse(x1,x2):
        return -20*np.log10(np.linalg.norm(x1-x2)/np.linalg.norm(x2)),-20*np.log10(np.linalg.norm(x1-x2,axis=(1,2))/np.linalg.norm(x2,axis=(1,2)))

    all_nmse = []
    Lambdas = []
    norm_data = np.linalg.norm(data)

    for mod in Models:
        print(mod)
        nmseloc = []
        nmseloc_ind = []
        lam = []
        model = load_model(mod)
        noise = np.random.randn(*data.shape)
        Xn = data + 10**(-SNRVal/20)*norm_data*noise/np.linalg.norm(noise)
        Lambda = 2*Lrange*(np.random.rand(npoints,model.anchorpoints.shape[0])-0.5)
        if Lambda0 is not None:
            Lambda += Lambda0
        Lambdas.append(Lambda)
        rec = _get_barycenter(Lambda,model=model)
        xrec = rec.data
        err = -20*np.log10(np.linalg.norm(xrec-Xn,axis=(1,2))/np.linalg.norm(Xn))
        err = err-np.min(err)
        err = 256*err/np.max(err)
        all_nmse.append(np.array(err))

        if model.anchorpoints.shape[0] == 2:
            fig = plt.figure(figsize =(14, 9))
            ax = plt.axes(projection ='3d')
            ax.scatter(Lambda[:,0],Lambda[:,1],err,c=err,cmap=mpl.cm.coolwarm)
            ax.view_init(elev=elev, azim=azim)

    return all_nmse,Lambdas

##############################################################################################################################################
# Unit tests - TO BE DONE
###############################################################################################################################################


##############################################################################################################################################
# Training with tuning
###############################################################################################################################################

def training_lightning_tune(XTrain,arg_IAE=None,arg_train=None,Xvalidation=None,checkmodel=False,tune_params=None):

    """
    CPUData : if true, keeps the data local and only transfer the batches to the GPU
    """

    import pytorch_lightning as pl
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
    from pytorch_lightning.callbacks import StochasticWeightAveraging
    from pytorch_lightning.loggers import TensorBoardLogger
    from ray import air, tune
    from ray.air import session
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.integration.pytorch_lightning import TuneReportCallback,TuneReportCheckpointCallback


    if torch.cuda.is_available():
        device = 'cuda'
        kwargs = {}
        gpus_per_trial = 1
    else:
        device = 'cpu'
        kwargs = {}
        gpus_per_trial = 0

    print("device USED: ",device)

    if device == 'cuda': # if GPU
        torch.backends.cudnn.benchmark = True

    if arg_train is None:
        arg_train = get_train_args()

    if arg_IAE is None:
        print("Please provide arguments for the IAE model")

    ###
    ### normalisation
    ###

    XTrain = torch.as_tensor(_normalize(XTrain,norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
    arg_IAE["anchorpoints"] = torch.as_tensor(_normalize(arg_IAE["anchorpoints"],norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
    data_loader = DataLoader(XTrain, batch_size=arg_train["batch_size"], shuffle=True, **kwargs)

    # Initialize the data loader

    if Xvalidation is not None:
        Xvalidation = torch.as_tensor(_normalize(Xvalidation,norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
        validation_loader = DataLoader(Xvalidation, batch_size=arg_train["batch_size"], shuffle=True, **kwargs)
    else:
        validation_loader = None

    # Tune config params

    def get_tuneconfig(tune_params=None):

        if tune_params is None:
            tune_params = {"lr":[1e-5,1e-1]}

        config= {}
        params=[]

        if "lr" in tune_params:
            config["lr"] =  tune.loguniform(tune_params["lr"][0], tune_params["lr"][0])
            params.append("lr")

        if "fsizefactor" in tune_params:
            config["fsizefactor"] = tune.randint(tune_params["fsizefactor"][0],tune_params["fsizefactor"][1])
            params.append("fsizefactor")

        if "nfilterfactor" in tune_params:
            config["nfilterfactor"] = tune.randint(tune_params["nfilterfactor"][0],tune_params["nfilterfactor"][1])
            params.append("nfilterfactor")

        if "rholatconfactor" in tune_params:
            config["rholatconfactor"] = tune.choice(tune_params["rholatconfactor"])
            params.append("rholatconfactor")

        print(config)
        print(params)

        return config,params

    def train_tune(config,arg_IAE=None,arg_train=None, num_epochs=1000, num_gpus=0,data_loader=None,validation_loader=None):
        model = IAE(input_arg=arg_IAE,arg_train=arg_train,config=config)
        model=model.to(device)
        trainer = pl.Trainer(
            enable_progress_bar=True,default_root_dir=arg_train["default_root_dir"],max_epochs=arg_train["max_epochs"],accumulate_grad_batches=arg_train["accumulate_grad_batches"],
            enable_checkpointing=arg_train["enable_checkpointing"],profiler=arg_train["profiler"],accelerator='gpu', devices=1,
            callbacks=[StochasticWeightAveraging(swa_lrs=1e-3,swa_epoch_start=2500),
                TuneReportCheckpointCallback(
                    metrics={
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                    },
                    filename="checkpoint",
                    on="validation_end")
                ])
        trainer.fit(model,data_loader,validation_loader)

    def tune_asha(num_samples=10, num_epochs=10, gpus_per_trial=0,arg_IAE=None,arg_train=arg_train,data_loader=None,validation_loader=None,config=None,parameter_columns=["lr"]):

        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=1,
            reduction_factor=2)

        reporter = CLIReporter(
            parameter_columns=parameter_columns,
            metric_columns=["loss", "mean_accuracy", "training_iteration"])

        train_fn_with_parameters = tune.with_parameters(train_tune,arg_IAE=arg_IAE,arg_train=arg_train,
                                                        num_epochs=num_epochs,
                                                        num_gpus=gpus_per_trial,data_loader=data_loader,validation_loader=validation_loader)
        resources_per_trial = {"cpu": 8, "gpu": gpus_per_trial}

        tuner = tune.Tuner(
            tune.with_resources(
                train_fn_with_parameters,
                resources=resources_per_trial
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=air.RunConfig(
                name="tune_asha",
                progress_reporter=reporter,
            ),
            param_space=config,
        )
        results = tuner.fit()

        print("Best hyperparameters found were: ", results.get_best_result().config)


    def tune_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0,arg_IAE=None,arg_train=arg_train,data_loader=None,validation_loader=None,config=None,parameter_columns=["lr"]):

        scheduler = PopulationBasedTraining(
            perturbation_interval=4,
            hyperparam_mutations=config)

        reporter = CLIReporter(
            parameter_columns=parameter_columns,
            metric_columns=["loss", "mean_accuracy", "training_iteration"])

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    train_tune,arg_IAE=arg_IAE,arg_train=arg_train,
                                                        num_epochs=num_epochs,
                                                        num_gpus=gpus_per_trial,data_loader=data_loader,validation_loader=validation_loader),
                resources={
                    "cpu": 1,
                    "gpu": gpus_per_trial
                }
            ),
            tune_config=tune.TuneConfig(
                metric="mean_accuracy",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=air.RunConfig(
                name="tune_pbr",
                progress_reporter=reporter,
            ),
            param_space=config,
        )
        results = tuner.fit()

        print("Best hyperparameters found were: ", results.get_best_result().config)

    tune_config,params = get_tuneconfig(tune_params=tune_params)
    tune_pbt(num_samples=10, num_epochs=arg_train["max_epochs"], gpus_per_trial=gpus_per_trial,arg_IAE=arg_IAE,arg_train=arg_train,data_loader=data_loader,validation_loader=validation_loader,parameter_columns=params,config=tune_config)

    #if arg_train["verb"]:
    #    print("Training step")
    #mycb = [StochasticWeightAveraging(swa_lrs=1e-3,swa_epoch_start=2500)]

    #trainer = pl.Trainer(callbacks=mycb,default_root_dir=arg_train["default_root_dir"],max_epochs=arg_train["max_epochs"],accumulate_grad_batches=arg_train["accumulate_grad_batches"],auto_scale_batch_size=arg_train["auto_scale_batch_size"],auto_lr_find=arg_train["auto_lr_find"],enable_checkpointing=arg_train["enable_checkpointing"],profiler=arg_train["profiler"])
    #out_train = trainer.fit(IAEmodel, data_loader) # Callback pour la validation

    #if Xvalidation is not None:
    #    if arg_train["verb"]:
    #        print("Validation step")
    #    out_val = trainer.validate(IAEmodel, validation_loader,ckpt_path="best",verbose=True)
    #else:
    #    out_val = None

    # Saving the model

    #save_model(iae,fname=arg_train["fname"])

    #return iae, out_train, out_val



#def bsp(x,model=None,fname=None,a0=None,Lambda0=None,epochs= 1000,stepsizeA=1e-3,stepsizeL=1e-3,LossOpt='l2',simplex=True,nonneg_weights=False,npoints=10,init='FI',LineSearch=True):
#    """
#    This is just an example code to see how the IAE model could be used as a generative model
#    """
#
#    if model is None:
#        model = load_model(fname)
#
#    PhiE,_ = model.encode(model.anchorpoints)
#    d = model.anchorpoints.shape[0]
#    b,tx,ty = x.shape
#
#    # Initialize Lambda0
#
#    loss_val = []
#
#    if a0 is None:
#        _,a0 = _normalize(x,norm=model.normalisation)
#    a = torch.tensor(a0.astype("float32"), requires_grad=True)
#
#    if Lambda0 is None:
#        if init == 'rand':
#            Lambda0 = QuickSampling(x,model,npoints=npoints,simplex=model.simplex,nonneg_weights=model.nonneg_weights,Lrange=1)[0]
#        elif init == 'FI':
#            Lambda0 = model.fast_interpolation(x,Amplitude=a0)["Lambda"][0].detach().numpy()
#    Lambda = torch.tensor(Lambda0.astype("float32"), requires_grad=True)
#
#    F = _loss(LossOpt=LossOpt)
#
#    x = torch.as_tensor(x.astype("float32"))
#
#    ####
#
#    def LossFunc(Lambda,get_rec=False):
#
##if nonneg_weights:
##    u = Lambda
##    for r in range(u.shape[0]):
##        u[r] = torch.as_tensor(P_circlel1_nneg(u[r].detach().numpy(),1)) # Not quite clean
##    Lambda.data = u
##elif simplex:
##    u = Lambda
##    for r in range(u.shape[0]):
##        u[r] = torch.as_tensor(P_circlel1(u[r].detach().numpy(),1)) # Not quite clean
##    Lambda.data = u
#
#        B = []
#        for r in range(model.NLayers):
#            B.append(torch.einsum('ik,kjl->ijl',Lambda, PhiE[model.NLayers-r-1]))
#
#        xrec = torch.einsum('ijk,i -> ijk',model.decode(B),a)
#
#        if get_rec:
#            return F(xrec, x),xrec
#        else:
#            return F(xrec, x)
#
#    ####
#
#    def LineSearch_Dicho(Lambda,t,g,LossF,niter=5):
#
#        t_m = t*0.1
#        t_M = t*2
#        val_m = LossF(Lambda - t_m*g).item()
#        val_M = LossF(Lambda - t_M*g).item()
#        val = LossF(Lambda - t*g).item()
#        val_in = LossF(Lambda).item()
#
#        for r in range(niter):
#
#            t1 = 0.5*(t_m+t)
#            val1 = LossF(Lambda - t1*g).item()
#
#            t2 = 0.5*(t_M+t)
#            val2 = LossF(Lambda - t2*g).item()
#
#            if val1 < np.min([val_m, val]):
#                t_M = t
#                t = t1
#                val_M = val
#                val = val1
#            elif val2 < np.min([val_m, val]):
#                t_m = t
#                t = t2
#                val_m = val
#                val = val2
#            else:
#                break
#
#        return t
#
#    lossv=1
#    H = torch.autograd.functional.hessian(LossFunc, Lambda).squeeze()
#    st=1./np.linalg.norm(H)
#    valmin = 1e32
#
#    for it in range(epochs):
#
#        g = torch.autograd.functional.jacobian(LossFunc, Lambda)
#
#        if np.mod(it,25)==0:
#        #if it <10:
#            if LineSearch:
#                H = torch.autograd.functional.hessian(LossFunc, Lambda).squeeze()
#                st = 1/np.linalg.norm(H)
#                st = LineSearch_Dicho(Lambda,st,g,LossFunc,niter=3)
#            else:
#                H = torch.autograd.functional.hessian(LossFunc, Lambda).squeeze()
#                st = 0.1*st+0.9/np.linalg.norm(H)
#            print(it,lossv,st)
#        else:
#            st = 0.99*st
#
#        Lambda = Lambda - st*g
#
#        loss,xrec = LossFunc(Lambda,get_rec=True)
#        lossv=loss.item()
#        loss_val.append(loss.item())
#
#        # Update a
#
#        #a.data = a.data - stepsizeA * a
#        #a.data = a.data*(a.data > 0)
#
#
#
#    return loss_val,Lambda.detach().numpy(),a,xrec.detach().numpy()

#def bsp(x,model=None,fname=None,a0=None,Lambda0=None,epochs= 1000,LossOpt='l2',simplex=True,nonneg_weights=False,npoints=10,init='FI',LineSearch=True):
#    """
#    This is just an example code to see how the IAE model could be used as a generative model
#    WITH BFGS - Clean it up + amplitude
#    """
#
#    from scipy.optimize import minimize
#
#    if model is None:
#        model = load_model(fname)
#
#    PhiE,_ = model.encode(model.anchorpoints)
#    d = model.anchorpoints.shape[0]
#    b,tx,ty = x.shape
#
#    # Initialize Lambda0
#
#    loss_val = []
#
#    if a0 is None:
#        _,a0 = _normalize(x,norm=model.normalisation)
#    a = torch.tensor(a0.astype("float32"), requires_grad=True)
#
#    if Lambda0 is None:
#        if init == 'rand':
#            Lambda0 = QuickSampling(x,model,npoints=npoints,simplex=model.simplex,nonneg_weights=model.nonneg_weights,Lrange=1)[0]
#        elif init == 'FI':
#            Lambda0 = model.fast_interpolation(x,Amplitude=a0)["Lambda"][0].detach().numpy()
#    Lambda = torch.tensor(Lambda0.astype("float32"), requires_grad=True)
#
#    F = _loss(LossOpt=LossOpt)
#
#    x = torch.as_tensor(x.astype("float32"))
#
#    ####
#
#    def LossFunc(Lambda,a,get_rec=False):
#
#        L=torch.as_tensor(Lambda)
#
#        B = []
#        for r in range(model.NLayers):
#            B.append(torch.einsum('ik,kjl->ijl',L, PhiE[model.NLayers-r-1]))
#
#        xrec = torch.einsum('ijk,i -> ijk',model.decode(B),a)
#
#        if get_rec:
#            return F(xrec, x),xrec
#        else:
#            return F(xrec, x)
#
#    ####
#
#    def Func(P):
#        P = P.reshape(b,-1)
#        a,L = P[:,0],P[:,1::]
#        return LossFunc(torch.as_tensor(L.astype('float32')),torch.as_tensor(a.astype('float32'))).item()
#
#    P = np.concatenate((a.detach().numpy().astype('float32').reshape(-1,1),Lambda.detach().numpy().astype('float32')),1).reshape(-1,)
#
#    constraints = []
#    if simplex:
#        def simp(P):
#            P = P.reshape(b,-1)
#            a,L = P[:,0],P[:,1::]
#            return np.sum(L,axis=1) - 1
#        constraints.append({'type':'eq', 'fun': simp})
#    if nonneg_weights:
#        def nneg(P):
#            return P
#        constraints.append({'type':'ineq', 'fun': nneg})
#
#    out = minimize(Func, P,  method='BFGS',options={'maxiter':epochs},constraints=constraints)
#
#    P = out.x.reshape(b,-1)
#    a,Lambda = P[:,0],P[:,1::]
#
#    return _get_barycenter(Lambda.reshape(b,-1),amplitude=a.squeeze(),model=model),Lambda,a
#
