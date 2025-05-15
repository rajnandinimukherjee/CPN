import torch
import pdb

from actions import ToyModelAction
from utils import CNtoR2N
import deformations as defs


def OnePointFn(i=0, j=0, action=ToyModelAction, pidx=0,
               deform=False, model=None, alphas=None, beta=4.5,
               batch_idx=slice(None), sampletype="train",
               pause=False, **kwargs):

    assert sampletype in ["train", "test",
                          "all"], "sampletype must be train, test or all"

    x = model.samples[sampletype]
    if sampletype == "train":
        x = x[batch_idx]

    if deform:
        X = CNtoR2N(x)
        Z = model.complexify(X, alphas)

        z = Z[..., ::2] + 1j * Z[..., 1::2]
        zconj = Z[..., ::2] - 1j * Z[..., 1::2]

        obs = z[:, pidx, i] * zconj[:, pidx, j]
        Sdiff = torch.exp(-action(z, beta, conj=zconj) + action(x, beta))
        detJ = model.detJac(X, alphas)

        obs *= Sdiff*detJ
    else:
        obs = x[:, pidx, i] * x.conj()[:, pidx, j]

    if pause:
        pdb.set_trace()

    return obs
