import torch
import pdb


def ToyModelAction(vecs, beta, conj=None):
    z, w = vecs[..., 0, :], vecs[..., 1, :]
    if conj is not None:
        zconj, wconj = conj[..., 0, :], conj[..., 1, :]
    else:
        zconj, wconj = z.conj(), w.conj()

    zdagw = torch.sum(zconj*w, dim=-1)
    wdagz = torch.sum(wconj*z, dim=-1)
    return -beta * zdagw * wdagz
