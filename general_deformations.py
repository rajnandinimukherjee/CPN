import pdb

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from actions import ToyModelAction
from deformations import HomogenousDeformations
from observables import OnePointFn
from plot_settings import plotparams
from utils import call_PDF

plt.rcParams.update(plotparams)


class AlphaNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim1=16, hidden_dim2=8):
        super(AlphaNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1, dtype=torch.double),
            nn.ReLU(),
            # nn.Linear(hidden_dim1, hidden_dim2, dtype=torch.double),
            # nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim1, output_dim, dtype=torch.double)
        )

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

        self.initial_var = obs(
            func='var', model=self.model, sampletype="all", **self.kwargs).item()
        self.target_exp = obs(
            func='exp', model=self.model, sampletype="all", **self.kwargs).item()

    def forward(self, alphas, **kwargs):
        return self.variance(alphas, **kwargs)

    def variance(self, alphas, batch_idx=None, sampletype="train"):
        return self.obs(func='var', deform=True, alphas=alphas,
                        model=self.model, batch_idx=batch_idx,
                        sampletype=sampletype, **self.kwargs)

    def expectation(self, alphas, batch_idx=None, sampletype="train"):
        return self.obs(func='exp', deform=True, alphas=alphas,
                        model=self.model, batch_idx=batch_idx,
                        sampletype=sampletype, **self.kwargs)


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

        self.train_var, self.train_exp = [], []
        self.test_var, self.test_exp = [], []

    def train(self, update=False):
        for epoch in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()

            batch_idx = torch.randperm(
                self.model.samples["train"].size(0))[:self.batch_size]
            loss = self.loss_fn(self.parnet, batch_idx=batch_idx)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0 and update:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

            self.train_var.append(loss.item())
            self.train_exp.append(
                self.loss_fn.expectation(self.parnet, batch_idx=batch_idx))
            self.test_var.append(self.loss_fn.variance(
                self.parnet, sampletype="test"))
            self.test_exp.append(self.loss_fn.expectation(
                self.parnet, sampletype="test"))

        self.train_var = torch.tensor(self.train_var)
        self.train_exp = torch.tensor(self.train_exp)
        self.test_var = torch.tensor(self.test_var)
        self.test_exp = torch.tensor(self.test_exp)

    def plot_training(self, fname='training.pdf', show=True,
                      plot_error=True):
        i, j = self.loss_fn.kwargs["i"], self.loss_fn.kwargs["j"]
        sub_str = f"{i+1}{j+1}"

        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(5, 8),
                               gridspec_kw={"height_ratios": [2, 1, 1]})

        ax[0].plot(range(self.epochs), self.train_exp,
                   label="train", c="tab:blue")
        ax[0].plot(range(self.epochs), self.test_exp,
                   label="test", c="tab:orange")
        ax[0].axhline(self.loss_fn.target_exp, c='k', label='undeformed')
        ax[0].set_ylabel(r'$\mathtt{Exp}\left[Q_{'+sub_str+r'}\right]$')

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

        ax[2].set_xlim([0, self.epochs])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

        call_PDF(fname, show=show)


N_EPOCHS = 450
BETA = 4.5
ACTION = ToyModelAction
OBS = OnePointFn
I, J, VARIDX = 0, 0, 0
BATCH_SIZE = 1000
TRAINING_SPLIT = 0.8

if __name__ == "__main__":

    file = h5py.File("CPN.h5", 'r')["configs"]
    burn = int(file["info"]["burn"][0])
    stepsize = int(file["info"]["stepsize"][0])
    samples = torch.tensor(
        file["vectors"][burn+1::stepsize*10],
        dtype=torch.complex128)
    N_conf = len(samples)

    train_indices = torch.randperm(N_conf)[:int(TRAINING_SPLIT*N_conf)]

    N = samples.shape[-1] - 1
    model = HomogenousDeformations(N, deftype="general")
    model.samples = {
        "all": samples,
        "train": samples[train_indices],
        "test": samples[~train_indices]
    }

    obskwargs = {
        "i": I,
        "j": J,
        "varidx": VARIDX,
        "action": ACTION,
        "beta": BETA,
    }

    loss_fn = VarianceLoss(model, samples, N, OnePointFn, **obskwargs)

    alphanet = AlphaNet(2*model.DOF, model.DOF)
    optimizer = optim.Adam(alphanet.parameters(), lr=1e-3)
    trainer = Trainer(model, alphanet, loss_fn,
                      optimizer, N_EPOCHS, BATCH_SIZE)

    trainer.train(update=False)
    print(f"{N_conf} configs, batch size {BATCH_SIZE}")
    vari, varf = loss_fn.initial_var, loss_fn.variance(
        alphanet, sampletype="all")
    print(f"Variance reduction {vari} -> {varf} ({
          (vari-varf)*100/vari}%)")

    stni = loss_fn.target_exp/(vari**0.5)
    stnf = loss_fn.expectation(alphanet, sampletype="all")/(varf**0.5)
    print(f"StN improvement {stni} -> {stnf} ({
          (stnf-stni)*100/stni}%)")
    trainer.plot_training()
