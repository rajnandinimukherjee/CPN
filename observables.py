import torch
import pdb

from actions import ToyModelAction
from utils import CNtoR2N


def OnePointFn(i=0, j=0, action=ToyModelAction, func='var', varidx=0,
               deform=False, model=None, alphas=None, beta=4.5,
               batch_idx=slice(None), sampletype="train",
               pause=False, **kwargs):

    assert func in ["var", "exp"], "func must be exp or var"
    assert sampletype in ["train", "test",
                          "all"], "sampletype must be train, test or all"

    x = model.samples[sampletype]
    if sampletype == "train":
        x = x[batch_idx]

    if deform:
        X = CNtoR2N(x)
        Y = model.deformx(X, alphas)
        Z = model.complexify(X, Y)

        z = Z[..., ::2] + 1j * Z[..., 1::2]
        zconj = Z[..., ::2] - 1j * Z[..., 1::2]

        obs = z[:, varidx, i-1] * zconj[:, varidx, j-1]
        Sdiff = torch.exp(-action(z, beta, conj=zconj) + action(x, beta))
        detJ = model.detJac(X, Y, alphas)

        obs *= Sdiff*detJ
        if func == "var":
            obs *= zconj[:, varidx, i-1] * z[:, varidx, j-1]
            # conjugate of Sdiff should be Sdiff (all real?)
            obs *= Sdiff*detJ.conj()

    else:
        obs = x.conj()[:, varidx, i-1] * \
            x[:, varidx, j-1]
        if func == "var":
            obs *= obs.conj()

    if pause:
        pdb.set_trace()
    return torch.mean(obs).real
