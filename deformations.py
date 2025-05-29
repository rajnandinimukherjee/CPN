import pdb

import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from torchdiffeq import odeint

from utils import R2NtoCN, CNtoR2N, Lambda
from SUN import SUN_generators


class FlowDeformations(nn.Module):
    name = "FlowDef"

    def __init__(self, N, T=10, deftype="general"):
        super(FlowDeformations, self).__init__()
        self.N = N
        self.T = T
        self.deftype = deftype
        self.DOF = 2*N

    def getFlow(alphas):
        def flow(t, Z_flat):
            Z = R2NtoCN(Z_flat)

            f = alphas(t, Z)
            Zf = torch.sum(Z*f, dim=-2, keepdim=True)
            Zsq = torch.sum(Z*Z, dim=-2, keepdim=True)
            overlap = Zf/Zsq

            # contraint preserving projection
            dZdt = f - overlap*Z
            return CNtoR2N(dZdt)
        return flow

    def complexify(self, X, alphas):
        Z0 = torch.complex(X, torch.zeros_like(X))  # shape: (1, N)
        # assert torch.allclose(torch.sum(Z0**2).real,
        #                      torch.tensor(1.0), atol=1e-5)
        Z0_flat = CNtoR2N(Z0)

        flow = self.getFlow(alphas)
        tspan = torch.linspace(0, self.T, 1)
        ZT = odeint(flow, Z0_flat, tspan)[-1]

        return ZT

    def detJac(self, X, alphas):
        batch_shape = X.shape[:-1]
        flat_X = X.reshape(-1, X.shape[-1])

        def Zreal(x):
            Z = self.complexify(x, alphas)
            return Z.real

        def Zimag(x):
            Z = self.complexify(x, alphas)
            return Z.imag

        Jreal = vmap(jacrev(Zreal))(flat_X)
        Jimag = vmap(jacrev(Zimag))(flat_X)

        J = Jreal + 1j*Jimag
        J = J.reshape(*batch_shape, X.shape[-1], X.shape[-1])

        detJ = torch.linalg.det(J)

        return torch.prod(detJ, dim=-1)


class SkewMatrixDeformations(nn.Module):
    name = "SkewMatDef"

    def __init__(self, N, deftype="general"):
        super(SkewMatrixDeformations, self).__init__()
        self.N = N
        self.deftype = deftype

        self.size = int(2*(N+1))
        self.DOF = int(self.size*(self.size-1)/2)
        self.skew_idx = torch.triu_indices(self.size, self.size, offset=1)
        self.flat_idx = self.skew_idx[0]*self.size + self.skew_idx[1]

    def __repr__(self):
        return self.name+"_cons" if self.deftype == "constant" \
            else self.name+"_gen"

    def build_A(self, alphas):
        batch_shape = alphas.shape[:-1]
        assert alphas.shape[-1] == self.DOF, "Alphas of incorrect size"

        A_flat = torch.zeros((*batch_shape, self.size**2), dtype=torch.double)
        flat_idx = self.flat_idx.unsqueeze(0).expand(
            *batch_shape, -1) if alphas.dim() > 1 else self.flat_idx
        A_flat = A_flat.scatter(-1, flat_idx, alphas)
        A = A_flat.view(*batch_shape, self.size, self.size)
        A = A - torch.transpose(A, -1, -2)
        return A

    def deformx(self, X, alphas):
        if not isinstance(alphas, torch.Tensor):
            a = alphas(X)
            A = self.build_A(a)
        else:
            A = self.build_A(alphas)
        Y = torch.einsum('...ij,...j->...i', A, X)
        return Y

    def complexify(self, X, alphas):
        Y = self.deformx(X, alphas)
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, alphas):
        Y = self.deformx(X, alphas)
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


class ProjectorDeformations(nn.Module):
    name = "ProjDef"

    def __init__(self, N, deftype="general"):
        super(ProjectorDeformations, self).__init__()
        self.N = N
        self.deftype = deftype

        self.size = int(2*(N+1))
        self.DOF = int(self.size*self.size)
        self.eye = torch.eye(self.size)

    def __repr__(self):
        return self.name+"_cons" if self.deftype == "constant" \
            else self.name+"_gen"

    def deformx(self, X, alphas):
        if not isinstance(alphas, torch.Tensor):
            a = alphas(X)
            B = a.view(*a.shape[:-1], self.size, self.size)
        else:
            assert alphas.shape[-1] == self.DOF, "Alphas of incorrect size"
            B = alphas.view(*alphas.shape[:-1], self.size, self.size)

        PX = torch.einsum('...i,...j->...ij', X, X)
        Id = self.eye.expand(*X.shape[:-1], self.size,
                             self.size) if X.dim() > 1 else self.eye
        A = torch.einsum('...ij, ...jk->...ik', (Id-PX), B)
        Y = torch.einsum('...ij,...j->...i', A, X)
        return Y

    def complexify(self, X, alphas):
        Y = self.deformx(X, alphas)
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, alphas):
        Y = self.deformx(X, alphas)
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


class TorusDeformations(nn.Module):
    name = "TorusDef"

    def __init__(self, N, deftype="constant"):
        super(TorusDeformations, self).__init__()
        self.N = N
        self.deftype = deftype
        self.DOF = N
        self.eye = torch.eye(2*(N+1)).to(torch.complex128)

        self.basis = SUN_generators(N+1, cartan=True)

    def __repr__(self):
        return self.name+"_cons" if self.deftype == "constant" \
            else self.name+"_gen"

    def deformx(self, X, alphas):
        assert alphas.shape[0] == self.DOF, "Alphas of incorrect size"

        alpha_H = torch.einsum(
            'i,ijk->jk', alphas.to(torch.complex128), self.basis)
        Y = torch.einsum('ij,...i->...j', CNtoR2N(
            1j * alpha_H, matrix=True), X.to(torch.complex128))
        return Y

    def complexify(self, X, alphas):
        Y = self.deformx(X, alphas)
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, alphas):
        Y = self.deformx(X, alphas)
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
    name = "HomogDef"

    def __init__(self, N, deftype="constant"):
        super(HomogenousDeformations, self).__init__()
        self.N = N
        self.deftype = deftype
        self.DOF = N+1
        self.eye = torch.eye(2*self.DOF)

        blocks = [torch.tensor([[0, 1], [-1, 0]])] * (N+1)
        self.omega = torch.block_diag(*blocks).to(torch.float64)

    def __repr__(self):
        return self.name+"_cons" if self.deftype == "constant" \
            else self.name+"_gen"

    def deformx(self, X, alphas):
        if not isinstance(alphas, torch.Tensor):
            A = torch.diag_embed(
                torch.repeat_interleave(alphas(X), repeats=2, dim=-1))
        else:
            assert alphas.shape[-1] == self.DOF, "Alphas of incorrect size"

            A = torch.diag_embed(
                torch.repeat_interleave(alphas, repeats=2, dim=-1))
        Y = torch.einsum('...ij,...j->...i', A@self.omega, X)
        return Y

    def complexify(self, X, alphas):
        Y = self.deformx(X, alphas)
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, alphas):
        Y = self.deformx(X, alphas)
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
