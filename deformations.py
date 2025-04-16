import pdb

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap

from utils import CNtoR2N, Lambda
from SUN import SUN_generators


class SkewMatrixDeformations(nn.Module):
    def __init__(self, N):
        super(SkewMatrixDeformations, self).__init__()
        self.N = N

    def deformx(self, X, alphas):
        dim = X.shape[-1]
        assert alphas.shape[0] == (dim)*(dim-1)/2, \
            "Alphas of incorrect size"

        A = torch.zeros((2*self.N+2, 2*self.N+2))
        indices = torch.triu_indices(2*self.N+2, 2*self.N+2, offset=1)
        A[indices[0], indices[1]] = alphas
        A = A-A.T
        Y = torch.einsum('ij,...j->...i', A, X)
        return Y

    def complexify(self, X, Y, alphas):
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, Y, alphas):
        if X.ndim == 1:
            L = Lambda(Y)
            Z = self.complexify(X, Y, alphas)

            J_real = torch.autograd.functional.jacobian(lambda X: Z.real, X)
            J_imag = torch.autograd.functional.jacobian(lambda X: Z.imag, X)

            J = J_real + 1j*J_imag
            return torch.linalg.det(J)/(L**2)
        else:
            return 0


class TorusDeformations(nn.Module):
    def __init__(self, N, deftype="constant"):
        super(TorusDeformations, self).__init__()
        self.N = N
        self.deftype = deftype
        self.DOF = N
        self.eye = torch.eye(2*(N+1)).to(torch.complex128)

        self.basis = SUN_generators(N+1, cartan=True)

    def deformx(self, X, alphas):
        assert alphas.shape[0] == self.DOF, \
            "Alphas of incorrect size"

        alpha_H = torch.einsum(
            'i,ijk->jk', alphas.to(torch.complex128), self.basis)
        Y = torch.einsum('ij,...i->...j', CNtoR2N(1j *
                         alpha_H, matrix=True), X.to(torch.complex128))
        return Y

    def complexify(self, X, Y, alphas):
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, Y, alphas):
        if self.deftype == "constant":
            L = Lambda(Y).to(torch.complex128)
            L2dim = L.unsqueeze(-1).unsqueeze(-1)
            A = CNtoR2N(1j*torch.einsum('i,ijk->jk',
                        alphas.to(torch.complex128),
                self.basis), matrix=True)

            J = L2dim * self.eye
            J -= torch.einsum('...i,...j->...ij', X,
                              X).to(torch.complex128)@A@A/L2dim
            J += 1j * A
            detJ = torch.linalg.det(J).real/(L**2)
            return torch.prod(detJ, dim=-1)


class HomogenousDeformations(nn.Module):
    def __init__(self, N, deftype="constant"):
        super(HomogenousDeformations, self).__init__()
        self.N = N
        self.deftype = deftype
        self.DOF = N+1
        self.eye = torch.eye(2*self.DOF)

        blocks = [torch.tensor([[0, 1], [-1, 0]])] * (N+1)
        self.omega = torch.block_diag(*blocks).to(torch.float64)

    def deformx(self, X, alphas):
        if not isinstance(alphas, torch.Tensor):
            A = torch.diag_embed(
                torch.repeat_interleave(alphas(X), repeats=2, dim=-1))
        else:
            assert alphas.shape[-1] == self.DOF, \
                "Alphas of incorrect size"

            A = torch.diag_embed(
                torch.repeat_interleave(alphas, repeats=2, dim=-1))
        Y = torch.einsum('...ij,...j->...i', A@self.omega, X)
        return Y

    def complexify(self, X, Y):
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, Y, alphas):
        if self.deftype == "constant":
            L = Lambda(Y)
            L2dim = L.unsqueeze(-1).unsqueeze(-1)
            A = torch.diag_embed(
                torch.repeat_interleave(alphas, repeats=2, dim=-1))
            A2 = torch.einsum('...ij, ...jk->...ik', A, A)

            J = L2dim * self.eye
            J += torch.einsum('...ij,...j,...k->...ik', A2, X, X)/L2dim
            J = J.to(torch.complex128)
            J -= 1j * (A@self.omega)
            detJ = torch.linalg.det(J)/(L**2)
        else:
            batch_shape = X.shape[:-1]
            flat_X = X.reshape(-1, X.shape[-1])

            if not isinstance(alphas, torch.Tensor):
                def Zreal(x):
                    a = alphas(x)
                    y = self.deformx(x, a)
                    return x*Lambda(y).unsqueeze(-1)

                def Zimag(x):
                    a = alphas(x)
                    y = self.deformx(x, a)
                    return y

                Jreal = vmap(jacrev(Zreal))(flat_X)
                Jimag = vmap(jacrev(Zimag))(flat_X)
            else:
                def Z(x):
                    y = self.deformx(x, alphas)
                    return self.complexify(x, y)

                Jreal = vmap(jacrev(lambda x: Z(x).real))(flat_X)
                Jimag = vmap(jacrev(lambda x: Z(x).imag))(flat_X)

            J = Jreal + 1j*Jimag
            J = J.reshape(*batch_shape, X.shape[-1], X.shape[-1])

            detJ = torch.linalg.det(J)/(Lambda(Y)**2)

        return torch.prod(detJ, dim=-1)
