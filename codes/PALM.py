import numpy as np
import torch
from codes.utils import projection_simplex_torch

def PALM(set_loader, modelIAE_list, alpha = 0.9, beta=0.9, itmax = 1000, eps = 1e-6,Amplitude = None, device='cpu',mode_test=False):
    X_est_list = []
    A_est_list = []
    itmax_list = []
    
    for data in set_loader: # For all mini-batches
        Y,A,X = data
        Y = Y.to(device)
        A = A.to(device)
        X = X.to(device)

        X_est_mb = torch.zeros(X.size(), device=device)
        A_est_mb = torch.zeros(A.size(), device=device)
        it_max_mb = torch.zeros(Y.size()[0], device=device)

        for j in range(Y.size()[0]): # For all samples in the current mini-batch
            X_est = torch.ones(X_est_mb[0].size(), device=device) / 2.
            A_est = torch.ones(A_est_mb[0].size(), device=device) / 3.

            X_est_prev = X_est.clone() # For the stopping criterion
            A_est_prev = A_est.clone()

            it = 0
            while(torch.norm(X_est-X_est_prev, p = 'fro')/torch.norm(X_est, p = 'fro') > eps or torch.norm(A_est-A_est_prev, p = 'fro')/torch.norm(A_est, p = 'fro') > eps or it < 2) and it < itmax:
                if it>0:
                    if mode_test:
                        print('Convergence rate:')
                        print(torch.norm(X_est-X_est_prev, p = 'fro')/torch.norm(X_est, p = 'fro'), torch.norm(A_est-A_est_prev, p = 'fro')/torch.norm(A_est, p = 'fro'))
                    X_est_prev = X_est.clone()
                    A_est_prev = A_est.clone()
                                    
                X_est = X_est + alpha * torch.einsum('jl,kl->jkl', A_est, (Y[j]-torch.einsum('jkl,jl->kl', X_est, A_est))) # gradient descent

                for i in np.arange(X_est.shape[0]): # (approximated) projection on manifold
                    with torch.no_grad():
                        PhiX,PhiE = modelIAE_list[i].encode(X_est[i][None,:].clone())
                        B,_ = modelIAE_list[i].interpolator(PhiX,PhiE)
                        X_est[i] = modelIAE_list[i].decode(B).squeeze(0)
                    
                A_est = A_est + beta * torch.einsum('ikl,ijkl->ijl', Y[j]-torch.einsum('ijkl,ijl->ikl', X_est, A_est), X_est) # gradient descent 
                # A_est = projection_simplex_torch(A_est.squeeze(-1), axis=1).unsqueeze(-1) # projection on simplex
                A_est = A_est * (A_est>0) # projection positivity

                it += 1


            X_est_mb[j] = X_est
            A_est_mb[j] = A_est
            it_max_mb[j] = it

        X_est_list.append(X_est_mb)
        A_est_list.append(A_est_mb)
        itmax_list.append(it_max_mb)

    return X_est_list, A_est_list, itmax_list

def PALM_LS(set_loader, modelIAE_list, alpha = 0.9, itmax = 1000, eps = 1e-6,Amplitude = None, device='cpu',mode_test=False):
    X_est_list = []
    A_est_list = []
    itmax_list = []
    
    for data in set_loader: # For all mini-batches
        Y,A,X = data
        Y = Y.to(device)
        A = A.to(device)
        X = X.to(device)

        X_est_mb = torch.zeros(X.size(), device=device)
        A_est_mb = torch.zeros(A.size(), device=device)
        it_max_mb = torch.zeros(Y.size()[0], device=device)

        for j in range(Y.size()[0]): # For all samples in the current mini-batch
            X_est = torch.ones(X_est_mb[0].size(), device=device) / 2.
            A_est = torch.ones(A_est_mb[0].size(), device=device) / 3.

            X_est_prev = X_est.clone() # For the stopping criterion
            A_est_prev = A_est.clone()

            it = 0
            while(torch.norm(X_est-X_est_prev, p = 'fro')/torch.norm(X_est, p = 'fro') > eps or torch.norm(A_est-A_est_prev, p = 'fro')/torch.norm(A_est, p = 'fro') > eps or it < 2) and it < itmax:
                if it>0:
                    if mode_test:
                        print('Convergence rate: ')
                        print(torch.norm(X_est-X_est_prev, p = 'fro')/torch.norm(X_est, p = 'fro'), torch.norm(A_est-A_est_prev, p = 'fro')/torch.norm(A_est, p = 'fro'))
                    X_est_prev = X_est.clone()
                    A_est_prev = A_est.clone()
                                    
                X_est = X_est + alpha * torch.einsum('jl,kl->jkl', A_est, (Y[j]-torch.einsum('jkl,jl->kl', X_est, A_est)))

                for i in np.arange(X_est.shape[0]):
                    with torch.no_grad():
                        PhiX,PhiE = modelIAE_list[i].encode(X_est[i][None,:].clone())
                        B,_ = modelIAE_list[i].interpolator(PhiX,PhiE)
                        X_est[i] = modelIAE_list[i].decode(B).squeeze(0)
                    
                A_est = torch.einsum('kl,kjl->jl', Y[j], torch.pinverse(X_est.squeeze()).unsqueeze(-1)) # Least-Squares
                # A_est = projection_simplex_torch(A_est.squeeze(-1), axis=1).unsqueeze(-1) # projection on simplex
                A_est = A_est * (A_est>0) # projection positivity

                it += 1


            X_est_mb[j] = X_est
            A_est_mb[j] = A_est
            it_max_mb[j] = it

        X_est_list.append(X_est_mb)
        A_est_list.append(A_est_mb)
        itmax_list.append(it_max_mb)

    return X_est_list, A_est_list, itmax_list


# def gPALM(set_loader, modelIAE_list, alpha = 0.9, itmax = 1000, eps = 1e-6,Amplitude = None, device='cpu',mode_test=False):
#     X_est_list = []
#     A_est_list = []
#     itmax_list = []
    
#     for data in set_loader: # For all mini-batches
#         Y,A,X = data
#         Y = Y.to(device)
#         A = A.to(device)
#         X = X.to(device)

#         X_est_mb = torch.zeros(X.size(), device=device)
#         A_est_mb = torch.zeros(A.size(), device=device)
#         it_max_mb = torch.zeros(Y.size()[0], device=device)

#         for j in range(Y.size()[0]): # For all samples in the current mini-batch
#             A_est = torch.ones(A_est_mb[0].size(), device=device) / 3.
#             lam_est = torch.ones(model_IAE.num_ap, device=device)/model_IAE.num_ap
#             A_est_prev = A_est.clone()
#             lam_est_prev = lam_est.clone()

#             it = 0
#             while(torch.norm(lam_est-lam_est_prev, p = 'fro')/torch.norm(lam_est, p = 'fro') > eps or torch.norm(A_est-A_est_prev, p = 'fro')/torch.norm(A_est, p = 'fro') > eps or it < 2) and it < itmax:
#                 if it>0:
#                     if mode_test:
#                         print(torch.norm(lam_est-lam_est_prev, p = 'fro')/torch.norm(lam_est, p = 'fro'), torch.norm(A_est-A_est_prev, p = 'fro')/torch.norm(A_est, p = 'fro'))
#                     A_est_prev = A_est.clone()
#                     lam_est_prev = lam_est.clone()
                                    
#                 for i, model_IAE in enumerate(modelIAE_list):
#                     with torch.no_grad():
#                         _,PhiE = model_IAE.encode(model_IAE.anchorpoints)

#                     def cost_fun(lam):
#                         B = []
#                         for r in range(model_IAE.NLayers):
#                             B.append(torch.einsum('k,kjl->jl', lam, PhiE[model_IAE.NLayers-r-1])[None,:])
#                         psi_lam = model_IAE.decode(B)
#                         return torch.norm(y-psi_lam, p = 'fro')**2

#         lam_copy = lam_est.clone().requires_grad_()
#         loss = cost_fun(lam_copy)
#         v = torch.autograd.grad(loss, lam_copy)[0]

#         lam_est = lam_est - alpha * v

                    
#                 A_est = torch.einsum('kl,kjl->jl', Y[j], torch.pinverse(X_est.squeeze()).unsqueeze(-1))

#                 it += 1


#             X_est_mb[j] = X_est
#             A_est_mb[j] = A_est
#             it_max_mb[j] = it

#         X_est_list.append(X_est_mb)
#         A_est_list.append(A_est_mb)
#         itmax_list.append(it_max_mb)

#     return X_est_list, A_est_list, itmax_list