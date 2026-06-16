import torch
from CPN_utils import call_PDF
import matplotlib.pyplot as plt
from plot_settings import plotparams
plt.rcParams.update(plotparams)


def get_epoch_sat(trainer, window_size=10, min_change=1e-4, min_idx=100):
    rel_changes = torch.abs(
        (trainer.test_var[:-1]-trainer.test_var[1:])/trainer.test_var[:-1])

    kernel = torch.ones(window_size)
    stable_mask = (rel_changes < min_change).float()
    convolved = torch.nn.functional.conv1d(
        stable_mask.view(1, 1, -1),  # input
        kernel.view(1, 1, -1),       # kernel
        padding=0
    ).view(-1)
    all_idx = (convolved == window_size).nonzero(as_tuple=True)[0]
    stable_idx = all_idx[all_idx >= min_idx]
    epochs = torch.arange(trainer.epochs)
    if len(stable_idx) > 0:
        epoch_sat = epochs[stable_idx[0] + window_size]
    else:
        epoch_sat = epochs[-1]
    return epoch_sat


def plot_training(trainer, fname="", show=True,
                  plot_label="", plot_error=True,):
    i, j = trainer.loss_fn.kwargs["i"], trainer.loss_fn.kwargs["j"]
    sub_str = f"{i+1}{j+1}"

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(3, 5),
                           gridspec_kw={"height_ratios": [1.5, 1, 1]})

    ax[0].plot(range(trainer.epochs), trainer.train_exp,
               label="train", c="tab:blue")
    ax[0].plot(range(trainer.epochs), trainer.test_exp,
               label="test", c="tab:orange")
    ax[0].axhline(trainer.loss_fn.target_exp, c='k', label='undeformed')
    ax[0].set_ylabel(
        r'$\mathrm{Re}\left[\mathtt{Exp}Q_{'+sub_str+r'}\right]$')

    if plot_error:
        ax[0].fill_between(range(trainer.epochs),
                           trainer.train_exp +
                           (trainer.train_var/trainer.batch_size)**0.5,
                           trainer.train_exp -
                           (trainer.train_var/trainer.batch_size)**0.5,
                           color="tab:blue", alpha=0.3)
        ax[0].fill_between(range(trainer.epochs),
                           trainer.test_exp +
                           (trainer.test_var/trainer.test_size)**0.5,
                           trainer.test_exp -
                           (trainer.test_var/trainer.test_size)**0.5,
                           color="tab:orange", alpha=0.3)
        ax[0].fill_between(range(trainer.epochs),
                           trainer.loss_fn.target_exp +
                           (trainer.loss_fn.initial_var/trainer.train_size)**0.5,
                           trainer.loss_fn.target_exp -
                           (trainer.loss_fn.initial_var/trainer.train_size)**0.5,
                           color="k", alpha=0.1)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=1.0)

    ax[1].plot(range(trainer.epochs), trainer.train_var, label="train")
    ax[1].plot(range(trainer.epochs), trainer.test_var, label="test")
    ax[1].axhline(trainer.loss_fn.initial_var, c='k', label='undeformed')
    ax[1].set_ylabel(r'$\mathtt{Var}\left[Q_{'+sub_str+r'}\right]$')

    StN_train = trainer.train_exp/(trainer.train_var**0.5)
    StN_test = trainer.test_exp/(trainer.test_var**0.5)
    ax[2].plot(range(trainer.epochs), StN_train, label='train')
    ax[2].plot(range(trainer.epochs), StN_test, label='test')
    ax[2].axhline(trainer.loss_fn.target_exp/(trainer.loss_fn.initial_var**0.5),
                  c='k', label='undeformed')
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel(r'$\mathtt{StN}\left[Q_{'+sub_str+r'}\right]$')

    ax[2].set_xlim([0, get_epoch_sat(trainer)])
    ax[0].set_ylim([trainer.loss_fn.target_exp*0.9,
                   trainer.loss_fn.target_exp*1.1])
    ax[1].set_ylim([trainer.loss_fn.initial_var*0.1,
                   trainer.loss_fn.initial_var*1.1])

    fig.text(0.95, 0.5, plot_label, va='center', ha='left', rotation=270)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    if fname == "":
        fname = f"plots/{trainer.name}.pdf"
    call_PDF(fname, show=show)
    print(f"plot saved to {fname}")
