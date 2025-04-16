import pdb
from collections.abc import Callable
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def toymodelS(vecs: np.ndarray, beta: np.float64,
              **kwargs) -> np.float64:
    """ action S = -beta |z^dagger w|^2 """

    z, w = vecs
    zdagw = np.sum(z.conj()*w, axis=-1)
    wdagz = np.sum(w.conj()*z, axis=-1)
    return -beta*(zdagw*wdagz).real


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
    """ basic MCMC for generating configs of z and w
    example usage:
    MCMC([np.array([randCPN(3), randCPN(3)])], iters=500000, burn=5000,
         stepsize=10, N=3, action=toymodelS, beta=4.5, save=True)
    """

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


def randSUN(N, repeat=1):
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
