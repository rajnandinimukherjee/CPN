import pdb
import numpy as np
from scipy.special import comb
from spherical_harmonics import SphericalHarmonics


def so_generators(dim):
    gens = []
    norm = 1 / np.sqrt(2)
    for i in range(dim):
        for j in range(i + 1, dim):
            T = np.zeros((dim, dim))
            T[i, j] = 1
            T[j, i] = -1
            gens.append(norm * T)
    return np.array(gens)


def numHarmonics(dim, deg):
    return int(comb(deg+dim-1, deg)-comb(deg+dim-3, deg-2))


def ensure_dims(X, dim):
    if X.ndim == 1:
        X = X[None, :]
    Xdim = X.shape[-1]
    assert Xdim == dim, f"input dims ({Xdim}) don't match harmonics ({
        dim})"
    return X


def gradY(Ylm, x, N_har, dim, h=1e-6):
    B = x.shape[0]

    grad = np.zeros((B, N_har, dim))
    eye = np.eye(dim)

    for d in range(dim):
        shift = h*eye[d]
        f_fwd = Ylm(x+shift)
        f_bwd = Ylm(x-shift)
        grad[:, :, d] = (f_fwd-f_bwd)/(2*h)

    return grad


def project_gradY(gradY, x):
    proj = np.sum(gradY*x[:, None, :], axis=2, keepdims=True)
    return gradY - proj*x[:, None, :]


class VectorHarmonics:

    def __init__(self, dim, lmax):
        self.dim, self.lmax = dim, lmax
        self.Ylm = SphericalHarmonics(dim, lmax)
        self.loadInfo()
        self.so_gens = so_generators(self.dim)
        self.N_params = self.N_har*(1+len(self.so_gens))

    def loadInfo(self):
        N_har = [numHarmonics(self.dim, deg) for deg in range(self.lmax)]
        assert len(self.Ylm) == sum(N_har)
        edges = np.cumsum([0]+N_har)
        self.N_har = sum(N_har)
        self.l_idx = np.array([np.arange(edges[deg], edges[deg+1])
                              for deg in range(self.lmax)], dtype=object)
        self.lapl_eigvals = np.zeros(self.N_har)
        for deg in range(self.lmax):
            self.lapl_eigvals[self.l_idx[deg]] = deg*(deg+self.dim-2)

    def scalarHarmonics(self, X):
        X = ensure_dims(X, self.dim)
        geom = X.shape[:-1]
        X_flat = X.reshape(-1, self.dim)
        return self.Ylm(X_flat).reshape(*geom, self.N_har)

    def gradY(self, X):
        X = ensure_dims(X, self.dim)
        geom = X.shape[:-1]
        X_flat = X.reshape(-1, self.dim)

        gradYlm = gradY(self.Ylm, X_flat, self.N_har, self.dim)
        proj = project_gradY(gradYlm, X_flat)
        return proj.reshape(*geom, self.N_har, self.dim)

    def __call__(self, X, params, gradY=None, trace_jac=True):
        assert np.prod(params.shape) == self.N_params, \
            f"need {self.N_params} to parameterise VSH"
        X = ensure_dims(X, self.dim)
        geom = X.shape[:-1]
        X_flat = X.reshape(-1, self.dim)

        gradY = gradY or self.gradY(X)
        geom = gradY.shape[:-2]
        gradY_flat = gradY.reshape(-1, self.N_har, self.dim)
        params = params.reshape((1+len(self.so_gens)), self.N_har)
        phi = np.zeros(shape=(gradY_flat.shape[0], self.dim))
        for deg in range(self.lmax):
            params_l = params[:, self.l_idx[deg]]
            grad = gradY[:, self.l_idx[deg], :]
            phi += np.einsum('n,bnd->bd', params_l[0], grad)
            if deg == 0:
                rot = np.einsum('m,mij->ij', params_l[1:, 0], self.so_gens)
                phi += np.einsum('ij,bj->bi', rot, X_flat)
            else:
                divfree = np.einsum('mij, bnj -> mbni', self.so_gens, grad)
                phi += np.einsum('mn, mbni-> bi',
                                 params_l[1:,], divfree)
        if trace_jac:
            aY = self.scalarHarmonics(X)*params[0, :]
            trJ = -np.sum(aY*self.lapl_eigvals, axis=-1)
            pdb.set_trace()
            return phi.reshape(*geom, self.dim), trJ

        return phi.reshape(*geom, self.dim)
