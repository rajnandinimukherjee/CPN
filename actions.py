import torch
import pdb


def ToyModelAction(fields, beta, conj=None):
    assert fields.shape[-2] == 2, "toy model only considers 2 fields"
    f1, f2 = fields[..., 0, :], fields[..., 1, :]
    f1conj = f1.conj() if conj is None else conj[..., 0, :]
    f2conj = f2.conj() if conj is None else conj[..., 1, :]

    f1dagf2 = torch.sum(f1conj*f2, dim=-1)
    f2dagf1 = torch.sum(f2conj*f1, dim=-1)
    return -beta * (f1dagf2 * f2dagf1)
