import torch
import pdb
import h5py
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchdiffeq import odeint


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

    def forward(self, t, X):
        t_ = t.expand(*X.shape[:-1], 1)
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
        ax[0].set_ylim([self.loss_fn.target_exp*0.9,
                       self.loss_fn.target_exp*1.1])
        ax[1].set_ylim([self.loss_fn.initial_var*0.1,
                       self.loss_fn.initial_var*1.1])

        fig.text(0.95, 0.5, plot_label, va='center', ha='left', rotation=270)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

        if fname == "":
            fname = f"plots/{self.name}.pdf"
        call_PDF(fname, show=show)
        print(f"plot saved to {fname}")


N = 3
ACTION = ToyModelAction
OBS = OnePointFn
TRAIN_SPLIT = 0.8


if __name__ == "__main__":
    file = h5py.File(f"CP{N}.h5", 'r')["configs"]
    burn = int(file["info"]["burn"][0])
    stepsize = int(file["info"]["stepsize"][0])
    samples = torch.tensor(
        file["vectors"][burn+1::stepsize*10],
        dtype=torch.complex128)
    N_conf = len(samples)

    train_indices = torch.randperm(N_conf)[:int(TRAIN_SPLIT*N_conf)]


def loss_fn(Z, target):
    return torch.mean((Z - target) ** 2)


def train_flow():
    func = AlphaNet(input_dim=4*(N+1)+1, output_dim=4*(N+1))
    trange = torch.linspace(0.0, 1.0, 10)

    # Initial condition Z0
    Z0 = torch.cat([Xi, torch.zeros_like(Xi)], dim=-1)

    # Target we want Z(T) to reach
    target = torch.cat([Xf, Yf], dim=-1)

    optimizer = optim.Adam(func.parameters(), lr=0.01)

    for itr in tqdm(range(1000), desc="epoch:", leave=False):
        optimizer.zero_grad()
        ZT = odeint(func, Z0, trange)
        loss = loss_fn(ZT[-1], target)  # Use final state at t=1
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            print(f"Iter {itr}: Loss = {loss.item():.4f}")

    return func
