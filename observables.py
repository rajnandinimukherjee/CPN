import torch
import pdb

from actions import ToyModelAction
from utils import CNtoR2N


def OnePointFn(i=0, j=0, action=ToyModelAction, func='var', varidx=0,
               deform=False, model=None, alphas=None, Ps=None,
               Norm=None, beta=4.5, batch_idx=None, sampletype="train",
               pause=False, **kwargs):

    if sampletype == "train":
        x = model.train_samples.detach()
    elif sampletype == "test":
        x = model.test_samples.detach()
    else:
        raise ValueError("sampletype must be train or test")

    if batch_idx is not None and sampletype == "train":
        x = x[batch_idx]

    if Ps is None or batch_idx is not None or sampletype == "test":
        Ps = torch.exp(-action(x, beta))
    if Norm is None or batch_idx is not None or sampletype == "test":
        Norm = torch.sum(Ps)

    if deform:
        X = CNtoR2N(x)
        Y = model.deformx(X, alphas)
        Z = model.complexify(X, Y, alphas)

        z = Z[..., ::2] + 1j * Z[..., 1::2]
        zconj = Z[..., ::2] - 1j * Z[..., 1::2]

        Oij = zconj[:, varidx, i-1] * z[:, varidx, j-1]
        Sdiff = torch.exp(-action(z, beta, conj=zconj) + action(x, beta))
        detJ = model.detJac(X, Y, alphas)
        if pause:
            pdb.set_trace()
        Oij *= Sdiff*detJ
    else:
        Oij = x.conj()[:, varidx, i-1] * \
            x[:, varidx, j-1]

    return (torch.sum(Oij * Ps) / Norm).real if func == 'exp' \
        else (torch.sum(Oij ** 2 * Ps) / Norm).real
