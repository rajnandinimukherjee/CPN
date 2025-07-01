import pdb

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import training_utils as tu
from actions import ToyModelAction
from argparsing import parseargs
from observables import OnePointFn


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


class FlowNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super(FlowNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        )
        self.hd = hidden_dim

    def forward(self, t, X):
        t_ = t.expand(*X.shape[:-1], 1).to(X.dtype)
        tX = torch.cat([t_, X], dim=-1)
        return self.network(tX)


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
        with tqdm(range(self.epochs)) as pbar:
            for epoch in pbar:
                self.optimizer.zero_grad()

                batch_idx = torch.randperm(
                    self.model.samples["train"].size(0))[:self.batch_size]
                loss = self.loss_fn(alphas=self.parnet,
                                    batch_idx=batch_idx,
                                    sampletype="train")
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f"loss: {loss.item():.2f}")
                if epoch % 10 == 0 and update:
                    print(f"Epoch {epoch}: Loss = {loss.item()}")

                # self.train_var.append(loss.item())
                # self.train_exp.append(self.loss_fn.expectation(
                #    alphas=self.parnet,
                #    batch_idx=batch_idx,
                #    sampletype="train"))
                # self.test_var.append(self.loss_fn.variance(
                #    alphas=self.parnet,
                #    sampletype="test"))
                # self.test_exp.append(self.loss_fn.expectation(
                #    alphas=self.parnet,
                #    sampletype="test"))

        # self.train_var = torch.tensor(self.train_var)
        # self.train_exp = torch.tensor(self.train_exp).real
        # self.test_var = torch.tensor(self.test_var)
        # self.test_exp = torch.tensor(self.test_exp).real


ACTION = ToyModelAction
OBS = OnePointFn

if __name__ == "__main__":
    args, deformation = parseargs()
    N = args.N

    file = h5py.File(f"CP{N}.h5", 'r')["configs"]
    burn = int(file["info"]["burn"][0])
    stepsize = int(file["info"]["stepsize"][0])
    samples = torch.tensor(
        file["vectors"][burn+1::stepsize],
        dtype=torch.complex128)
    N_conf = len(samples)

    train_indices = torch.randperm(N_conf)[:int(args.split*N_conf)]

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

    netargs = {
        "input_dim": model.input_dim,
        "output_dim": model.DOF,
        "hidden_dim": args.hidden_dim
    }
    parnet = FlowNet(**netargs) if args.deftype == "FlowDef"\
        else AlphaNet(**netargs)

    optimizer = optim.Adam(parnet.parameters(), lr=1e-3)
    trainer = Trainer(model, parnet, loss_fn,
                      optimizer, args.epochs, args.batch_size)

    trainer.train(update=args.update)
    label = f"N = {N}, beta = {args.beta}, Nconf = {
        N_conf}, batch_size = {args.batch_size}, split = {args.split}"
    tu.plot_training(trainer, plot_label=label, show=False)
