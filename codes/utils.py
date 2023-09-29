import numpy as np
import torch


def generate_dataset_mixture(model_IAE_list, nb_samples, noise_level=[1e-3, 1e-1], sigma_a = 0., eps=0.):
    ''' generate mixtures Y = AX + n where X belongs to a malifold learned by model_IAE 
        and n is a Gaussian noise
        model_IAE_list: provide the IEA models to sample sources components X
        noise_level: range of standard deviation of the gaussian noise
        sigma_a: variability of relative scaling between components
        eps: variability of global scaling
    '''

    X = []
    for model_IAE in model_IAE_list:
        if model_IAE.mean_lambda:
            lam = torch.rand((nb_samples, model_IAE.num_ap), device=model_IAE.device).repeat(model_IAE.NLayers,1,1)
        else:
            lam = torch.rand((model_IAE.NLayers, nb_samples, model_IAE.num_ap), device=model_IAE.device)
        lam = torch.einsum('ijk,ij->ijk', lam, 1. / torch.sum(lam, 2)) 
        X.append(model_IAE.get_barycenter(lam).detach().numpy())
    X = np.stack(X).transpose((1,0,2,3))

    if sigma_a=="inf": # a random [0,1]
        A = np.random.rand(X.shape[0],X.shape[1],X.shape[3]) / len(model_IAE_list) * 2.
    else:
        A = 1 / len(model_IAE_list) + sigma_a * np.random.randn(X.shape[0],X.shape[1],X.shape[3])
    A = projection_simplex(A[:,:,0], axis=1)[:,:,None]   

    if not eps==0.:
        A = np.einsum('ijl,i->ijl', A, ((1.+eps)*np.ones(X.shape[0]))**((np.random.rand(X.shape[0]))*2-1))

    sigma = 10 ** (np.random.rand(X.shape[0]) * (np.log10(noise_level[1]/noise_level[0])) + np.log10(noise_level[0]))
    Y = np.einsum('ijkl, ijl->ikl', X, A) + np.einsum('ijk,i->ijk', np.random.randn(X.shape[0],X.shape[2],X.shape[3]), sigma)
    return Y, A, X


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

def projection_simplex_torch(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U,_ = torch.sort(V, axis=1,descending=True)
        z = torch.ones(len(V), device=V.device) * z
        cssv = torch.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = torch.arange(n_features).to(V.device) + 1
        cond = U - cssv / ind > 0
        rho = torch.count_nonzero(cond, axis=1)
        theta = cssv[torch.arange(len(V)), rho - 1] / rho
        return torch.maximum(V - theta.unsqueeze(-1), torch.tensor(0))

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()