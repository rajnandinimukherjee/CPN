import argparse
import pdb

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import deformations as defs
from actions import ToyModelAction
from observables import OnePointFn
from plot_settings import plotparams
from utils import call_PDF

plt.rcParams.update(plotparams)


class AlphaNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super(AlphaNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        )
        self.hd = hidden_dim

    def forward(self, X):
        return self.network(X)


class VarianceLoss(nn.Module):

    def __init__(self, model, samples, N, obs, **kwargs):
        super(VarianceLoss, self).__init__()
        self.model = model
        self.samples = samples
        self.N = N
        self.obs = obs
        self.kwargs = kwargs

        self.initial_var = self.variance(deform=False, sampletype="all")
        self.target_exp = self.expectation(deform=False, sampletype="all").real

    def get_obs(self, alphas=None, batch_idx=None,
                deform=True, sampletype="train"):
        return self.obs(deform=deform, alphas=alphas,
                        model=self.model, batch_idx=batch_idx,
                        sampletype=sampletype, **self.kwargs)

    def forward(self, **kwargs):
        return self.variance(**kwargs)

    def variance(self, **kwargs):
        return self.get_obs(**kwargs).var()

    def expectation(self, **kwargs):
        return self.get_obs(**kwargs).mean()


class Trainer:

    def __init__(self, model, parnet, loss_fn, optimizer, epochs, batch_size):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_size = self.model.samples["train"].size(0)
        self.test_size = self.model.samples["test"].size(0)
        self.parnet = parnet
        self.name = f"{self.model}_hdim{self.parnet.hd}"

        self.train_var, self.train_exp = [], []
        self.test_var, self.test_exp = [], []

    def train(self, update=False):
        for epoch in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()

            batch_idx = torch.randperm(
                self.model.samples["train"].size(0))[:self.batch_size]
            loss = self.loss_fn(alphas=self.parnet,
                                batch_idx=batch_idx, sampletype="train")
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0 and update:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

            self.train_var.append(loss.item())
            self.train_exp.append(self.loss_fn.expectation(
                alphas=self.parnet,
                batch_idx=batch_idx,
                sampletype="train"))
            self.test_var.append(self.loss_fn.variance(
                alphas=self.parnet,
                sampletype="test"))
            self.test_exp.append(self.loss_fn.expectation(
                alphas=self.parnet,
                sampletype="test"))

        self.train_var = torch.tensor(self.train_var)
        self.train_exp = torch.tensor(self.train_exp).real
        self.test_var = torch.tensor(self.test_var)
        self.test_exp = torch.tensor(self.test_exp).real

    def get_epoch_sat(self, window_size=10, min_change=1e-4, min_idx=100):
        rel_changes = torch.abs(
            (self.test_var[:-1]-self.test_var[1:])/self.test_var[:-1])

        kernel = torch.ones(window_size)
        stable_mask = (rel_changes < min_change).float()
        convolved = torch.nn.functional.conv1d(
            stable_mask.view(1, 1, -1),  # input
            kernel.view(1, 1, -1),       # kernel
            padding=0
        ).view(-1)
        all_idx = (convolved == window_size).nonzero(as_tuple=True)[0]
        stable_idx = all_idx[all_idx >= min_idx]
        epochs = torch.arange(self.epochs)
        if len(stable_idx) > 0:
            epoch_sat = epochs[stable_idx[0] + window_size]
        else:
            epoch_sat = epochs[-1]
        return epoch_sat

    def plot_training(self, fname="", show=True,
                      plot_label="", plot_error=True,):
        i, j = self.loss_fn.kwargs["i"], self.loss_fn.kwargs["j"]
        sub_str = f"{i+1}{j+1}"

        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(3, 5),
                               gridspec_kw={"height_ratios": [1.5, 1, 1]})

        ax[0].plot(range(self.epochs), self.train_exp,
                   label="train", c="tab:blue")
        ax[0].plot(range(self.epochs), self.test_exp,
                   label="test", c="tab:orange")
        ax[0].axhline(self.loss_fn.target_exp, c='k', label='undeformed')
        ax[0].set_ylabel(
            r'$\mathrm{Re}\left[\mathtt{Exp}Q_{'+sub_str+r'}\right]$')

        if plot_error:
            ax[0].fill_between(range(self.epochs),
                               self.train_exp +
                               (self.train_var/self.batch_size)**0.5,
                               self.train_exp -
                               (self.train_var/self.batch_size)**0.5,
                               color="tab:blue", alpha=0.3)
            ax[0].fill_between(range(self.epochs),
                               self.test_exp +
                               (self.test_var/self.test_size)**0.5,
                               self.test_exp -
                               (self.test_var/self.test_size)**0.5,
                               color="tab:orange", alpha=0.3)
            ax[0].fill_between(range(self.epochs),
                               self.loss_fn.target_exp +
                               (self.loss_fn.initial_var/self.train_size)**0.5,
                               self.loss_fn.target_exp -
                               (self.loss_fn.initial_var/self.train_size)**0.5,
                               color="k", alpha=0.1)

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=1.0)

        ax[1].plot(range(self.epochs), self.train_var, label="train")
        ax[1].plot(range(self.epochs), self.test_var, label="test")
        ax[1].axhline(self.loss_fn.initial_var, c='k', label='undeformed')
        ax[1].set_ylabel(r'$\mathtt{Var}\left[Q_{'+sub_str+r'}\right]$')

        StN_train = self.train_exp/(self.train_var**0.5)
        StN_test = self.test_exp/(self.test_var**0.5)
        ax[2].plot(range(self.epochs), StN_train, label='train')
        ax[2].plot(range(self.epochs), StN_test, label='test')
        ax[2].axhline(self.loss_fn.target_exp/(self.loss_fn.initial_var**0.5),
                      c='k', label='undeformed')
        ax[2].set_xlabel('epochs')
        ax[2].set_ylabel(r'$\mathtt{StN}\left[Q_{'+sub_str+r'}\right]$')

        ax[2].set_xlim([0, self.get_epoch_sat()])
        ax[0].set_ylim([self.loss_fn.target_exp*0.8,
                       self.loss_fn.target_exp*1.2])
        ax[1].set_ylim([0, self.loss_fn.initial_var*1.1])

        fig.text(0.95, 0.5, plot_label, va='center', ha='left', rotation=270)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

        if fname == "":
            fname = f"plots/{self.name}.pdf"
        call_PDF(fname, show=show)
        print(f"plot saved to {fname}")


ACTION = ToyModelAction
OBS = OnePointFn
def_names = ["HomogDef", "TorusDef", "ProjDef", "SkewMatDef", "FlowDef"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="CP(N) dimension", type=int, default=3)
    parser.add_argument("--i", help="OnePointFn element i",
                        type=int, default=0)
    parser.add_argument("--j", help="OnePointFn element j",
                        type=int, default=0)
    parser.add_argument(
        "--pidx", help="particle idx (0 | 1)", type=int, default=0)
    parser.add_argument(
        "--beta", help="ToyModelAction parameter beta", type=float, default=4.5)
    parser.add_argument("--epochs", help="epochs", type=int, default=1000)
    parser.add_argument("--batch_size", help="batch size",
                        type=int, default=1000)
    parser.add_argument("--split", help="training to test data split ratio",
                        type=float, default=0.8)
    parser.add_argument("--deftype", help=f"constant deformation type ({'|'.join(def_names)})",
                        type=str, default="HomogDef")
    parser.add_argument("--hidden_dim", help=f"alphanet hidden layer dimension",
                        type=int, default=16)
    args = parser.parse_args()

    assert args.deftype in def_names, f"Deformation type not recognized, choose from {', '.join(def_names)}"
    # ================================================================================
    N = args.N

    file = h5py.File(f"CP{N}.h5", 'r')["configs"]
    burn = int(file["info"]["burn"][0])
    stepsize = int(file["info"]["stepsize"][0])
    samples = torch.tensor(
        file["vectors"][burn+1::stepsize*10],
        dtype=torch.complex128)
    N_conf = len(samples)

    train_indices = torch.randperm(N_conf)[:int(args.split*N_conf)]

    if args.deftype == "TorusDef":
        deformation = defs.TorusDeformations
    elif args.deftype == "ProjDef":
        deformation = defs.ProjectorDeformations
    elif args.deftype == "SkewMatDef":
        deformation = defs.SkewMatrixDeformations
    elif args.deftype == "FlowDef":
        deformation = defs.FlowDeformations
    else:
        deformation = defs.HomogenousDeformations

    model = deformation(N, deftype="general")
    model.samples = {
        "all": samples,
        "train": samples[train_indices],
        "test": samples[~train_indices]
    }

    obskwargs = {
        "i": args.i,
        "j": args.j,
        "pidx": args.pidx,
        "action": ACTION,
        "beta": args.beta,
    }

    loss_fn = VarianceLoss(model, samples, N, OnePointFn, **obskwargs)

    alphanet = AlphaNet(2*(N+1), model.DOF, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(alphanet.parameters(), lr=1e-3)
    trainer = Trainer(model, alphanet, loss_fn,
                      optimizer, args.epochs, args.batch_size)

    trainer.train(update=False)
    label = f"N = {N}, beta = {args.beta}, Nconf = {N_conf}, batch_size = {args.batch_size}, split = {args.split}"
    trainer.plot_training(plot_label=label, show=False)
