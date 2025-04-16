import os

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages


def CNtoR2N(x, matrix=False):
    if matrix:
        assert x.shape[-1] == x.shape[-2], \
            "input must be square matrix in last 2 dims"
        new_vec = torch.zeros(
            x.shape[:-2]+(2*x.shape[-2], 2*x.shape[-1]),
            dtype=x.dtype)
        new_vec[..., 0::2, 0::2] = x.real
        new_vec[..., 0::2, 1::2] = -x.imag
        new_vec[..., 1::2, 0::2] = x.imag
        new_vec[..., 1::2, 1::2] = x.real
    else:
        new_vec = torch.zeros(
            x.shape[:-1] + (2 * x.shape[-1],), dtype=torch.double)
        new_vec[..., 0::2], new_vec[..., 1::2] = x.real, x.imag
    return new_vec


def R2NtoCN(X):
    return X[..., 0::2] + 1j * X[..., 1::2]


def Lambda(Y):
    # return (1 + torch.linalg.norm(Y, axis=-1)**2)**0.5
    return (1+torch.einsum('...i,...i->...', Y.conj(), Y))**0.5


def call_PDF(filename, show=True):
    """ plots matplotlib graphics to pdfs, saves and opens file"""

    pdf = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format="pdf")
    pdf.close()
    plt.close("all")
    if show:
        os.system("open " + filename)
