import pdb
from collections.abc import Callable
from typing import List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


def toymodelS(vecs: np.ndarray, beta: np.float64,
              **kwargs) -> np.float64:
    """ action S = -beta |z^dagger w|^2 """

    z, w = vecs
    zconj = kwargs['conj'][0] if 'conj' in kwargs else z.conj()
    return -beta*np.abs(np.einsum('...i,...i->...', zconj, w))**2


def metropolis(samples: List, action: Callable,
               N: int = 3, **actionkwargs) -> None:
    """ metropolis accept/reject step """

    old_vecs = samples[-1]

    rand_rotations = randSUN(N+1, repeat=old_vecs.shape[0])
    new_vecs = np.einsum('mij, mj->mi', rand_rotations, old_vecs)

    deltaS = action(new_vecs, **actionkwargs) -\
        action(old_vecs, **actionkwargs)

    if np.random.rand() < min(1.0, np.exp(-deltaS)):
        samples.append(new_vecs)
        return 1
    else:
        samples.append(old_vecs)
        return 0


def MCMC(samples: List, iters: int = 10000,
         burn: int = 1000, stepsize: int = 1,
         save: bool = False, fname: str = 'CPN.h5',
         **kwargs) -> np.ndarray:
    """ basic MCMC for generating configs of z and w """

    accepts = 0
    for idx in tqdm(range(iters), desc="iterations"):
        accepts += metropolis(samples, **kwargs)

    print(f'acceptance rate: {
          accepts}/{iters}={np.around(accepts*100/iters, 1)}%')
    if save:
        file = h5py.File(fname, 'a')
        if "configs" in file:
            del file["configs"]
        grp = file.create_group("configs")
        grp.create_dataset("vectors", data=np.array(samples))
        info = grp.create_group("info")
        info.create_dataset("iterations", data=[iters])
        info.create_dataset("burn", data=[burn])
        info.create_dataset("stepsize", data=[stepsize])
        file.close()
        print(f"CPN configs saved to {fname}")
    return np.array(samples)[burn+1::stepsize]


def OnePointFn(i: int, j: int, action: Callable, N: int = 3,
               varidx: int = 0, deform=False, run: bool = True,
               fname: str = 'CPN.h5', show=False, **kwargs) -> None:
    """ calculates expectation value and variance of CP(N) one-pt fn O_{ij} """

    if run:
        samples = MCMC([np.array([randCPN(N), randCPN(N)])], iters=50000,
                       burn=5000, stepsize=100, N=N, action=action, **kwargs)
    else:
        file = h5py.File(fname, 'r')["configs"]
        burn = int(file["info"]["burn"][0])
        stepsize = int(file["info"]["stepsize"][0])
        samples = np.array(file["vectors"][:])[burn+1::stepsize]
        print(f"CPN configs loaded from {fname}")

    print(f'O_{i}{j} sampled from {len(samples)} configurations')

    iters = samples.shape[0]
    Ps = np.array([np.exp(-action(samples[idx], **kwargs))
                   for idx in range(iters)])
    Z = np.sum(Ps)

    Oij = samples.conj()[:, varidx, i-1]*samples[:, varidx, j-1]
    Oij_exp = np.sum(Oij*Ps)/Z
    Oij_var = np.sum(Oij**2*Ps)/Z

    print(f'<Oij>={Oij_exp.real}, varOij={Oij_var.real}')

    if deform:
        alphas = kwargs['alphas']
        samples_x = samples.copy()
        samples_y = deformx(samples_x, alphas)
        samples_z, samples_zconj = getz(samples_x, samples_y)

        Qij = samples_zconj[:, varidx, i-1] *\
            samples_z[:, varidx, j-1]
        detjacobians = np.array([detJac(
            samples_x[:, varidx], samples_y[:, varidx], alphas, N=N)
            for varidx in range(samples.shape[1])]).swapaxes(0, 1)
        Qij *= np.prod(detjacobians, axis=-1)
        Qij *= np.array([np.exp(
            -action(samples_z[idx], conj=samples_zconj[idx], **kwargs)
            + action(samples_x[idx], **kwargs))
            for idx in range(iters)])

        Qij_exp = np.sum(Qij*Ps)/Z
        Qij_var = np.sum(Qij**2*Ps)/Z
        print(f'<Qij>={Qij_exp.real}, varQij={Qij_var.real}')


def randSUN(N: int, repeat: int = 1) -> np.ndarray:
    """ generates random elements of the SU(N)
    group in NxN matrix representation """

    U = np.random.randn(repeat, N, N) + 1j*np.random.randn(repeat, N, N)
    Q, R = np.linalg.qr(U, mode='complete')

    # enforce unitarity by making diag elements of R have norm 1
    D = np.exp(-1j*np.angle(np.einsum('...ii->...i', R)))[:, None, :]
    U = Q*D

    # make determinant 1
    detU = np.linalg.det(U)
    U /= detU[:, None, None]**(1/N)
    return U if repeat > 1 else U[0]


def R2NtoCN(x: np.ndarray) -> np.ndarray:
    """ forms complex vec C^N from real vec R^2N"""

    return x[..., 0::2]+1j*x[..., 1::2]


def CNtoR2N(x: np.ndarray) -> np.ndarray:
    """ forms real vec R^2N from complex vec C^N"""

    new_vec = np.zeros(shape=x.shape[:-1]+(2*x.shape[-1],), dtype=np.float64)
    new_vec[..., 0::2], new_vec[..., 1::2] = x.real, x.imag
    return new_vec


def randSN(N: int) -> np.ndarray:
    """ generates a random variable from S^N set
        in form of a normalised R^{N+1} vector """

    SN = np.random.randn(N+1)
    return SN/np.linalg.norm(SN)


def randCPN(N: int) -> np.ndarray:
    """ generates a random CP(N) vector """

    return R2NtoCN(randSN(2*N+1))


def complexdot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ computes dot product for complex variables """

    assert x.shape[-1] == y.shape[-1], \
        "input vectors must have same length of last axis"
    return np.einsum('...i,...i->...', x.conj(), y)

# ========== for deformed variables ====================================


def deformx(x: np.ndarray, alphas: np.ndarray, N: int = 3) -> np.ndarray:
    """ from x generates homogenously deformed y(x) = alpha Omega x """

    X = CNtoR2N(x)
    Y = np.einsum('ij, ...j->...i', alphaOmega(alphas, N+1), X)
    assert np.allclose(np.einsum('...i,...i->...', X, Y), 0), '<X,Y>!=0'
    return R2NtoCN(Y)


def getz(x: np.ndarray, y: np.ndarray, N: int = 3)\
        -> Tuple[np.ndarray, np.ndarray]:
    """ given x, y, return z and conjugate of z """

    X, Y = CNtoR2N(x), CNtoR2N(y)
    LX = X*Lambda(Y)[..., np.newaxis] if x.size > (2*N+2) else X*Lambda(Y)
    Z = LX + 1j*Y
    assert np.allclose(complexdot(LX, LX) - complexdot(Y, Y), 1.0), \
        '||LX||^2 - ||Y||^2 != 1'
    assert np.allclose(np.sum(Z**2, axis=-1), 1.0), 'Sum_k Z_k^2 != 1'

    z = Z[..., ::2] + 1j*Z[..., 1::2]
    zconj = Z[..., ::2] - 1j*Z[..., 1::2]

    return z, zconj


def alphaOmega(alphas: torch.Tensor, N: int, **kwargs) -> np.ndarray:
    """ produces homogenous deformation matrix alpha*Omega """

    omega = torch.zeros(2*N, 2*N)
    for mu in range(N):
        omega[2*mu:(2*mu+1)+1, 2*mu:(2*mu+1)+1] = alphas[mu] *\
            torch.tensor([[0, -1], [1, 0]])
    return omega.detach().numpy()


def Lambda(y: np.ndarray) -> np.float64:
    """ computes lambda from y vector """

    return (1+np.linalg.norm(y, axis=-1)**2)**0.5


def detJac(x: np.ndarray, y: np.ndarray, alphas: torch.Tensor,
           N: int = 3) -> np.ndarray:
    """ computes homogenous deformation jacobian """

    X, Y = CNtoR2N(x), CNtoR2N(y)
    L = Lambda(Y)
    AX = ((X*np.repeat(alphas, 2)).T*(1/L)).T

    J = L[..., np.newaxis, np.newaxis]*np.identity(2*N+2)
    J += (np.einsum('...a,...b->...ab', AX, AX).T*(1/L)).T
    J = J.astype('complex128')
    J -= 1j*alphaOmega(alphas, N+1)
    return np.linalg.det(J).real/(L**2)
