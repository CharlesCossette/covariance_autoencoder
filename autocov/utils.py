import torch
import matplotlib.pyplot as plt
import seaborn as sns


def tril_to_vec(x: torch.Tensor):
    """
    Converts a batch lower triangular matrix to a vector
    """
    return x[:, torch.tril(torch.ones(x.size(1), x.size(2))).bool()]


def vec_to_tril(x: torch.Tensor, size=45, offset=0):
    """
    Converts a batch vector to a lower triangular matrix
    """
    c = torch.zeros(x.size(0), size, size)
    c[:, torch.tril(torch.ones(size, size), diagonal=offset).bool()] = x
    return c


def tril_to_sym(x: torch.Tensor):
    """
    Converts a batch lower triangular matrix to a symmetric matrix.
    """
    out = x + x.transpose(1, 2)
    diagonal = torch.diagonal(out, dim1=1, dim2=2)
    out = out - torch.diag_embed(diagonal) / 2
    return out



def covariance_to_cholvec(covariance: torch.Tensor, size=45):
    """
    Converts a covariance matrix to a vector of the
    lower triangular part of the Cholesky decomposition
    """
    chol = torch.linalg.cholesky(covariance)
    return chol[:, torch.tril(torch.ones(size, size)).bool()]


def cholvec_to_covariance(cholvec: torch.Tensor, size=45):
    """
    Converts a vector of the lower triangular part of the Cholesky decomposition
    to a covariance matrix
    """
    c = torch.zeros(cholvec.shape[0], size, size)
    c[:, torch.tril(torch.ones(size, size)).bool()] = cholvec
    return c @ c.transpose(1, 2)


def plot_covariances(cov_true: torch.Tensor, cov_hat: torch.Tensor, num=None, cmap="RdBu", vmax=None):
    """
    Plots true vs reconstructed covariance matrices.

    Parameters
    ----------
    cov_true : torch.Tensor with shape [batch x N x N]
        true covariance matrices that will show up on top row
    cov_hat : torch.Tensor with shape [batch x N x N]
        reconstructed covariance matrices that will show up on bottom row
    num : int, optional
        number of matrices to plot, by default None. If None, plots all matrices.
        If an int, this will be uniformly linspaced from the batch.
    """
    if num is None:
        idx = torch.arange(cov_true.shape[0])
    else:
        idx = torch.linspace(0, cov_true.shape[0] - 1, num).long()

    num = len(idx)

    fig, axs = plt.subplots(2, num, figsize=(num * 3+4, 5))
    cov_true_plot = cov_true[idx]

    if vmax is None:
        vmax = cov_true_plot.max()
        vmin = cov_true_plot.min()
        vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax
    # Plot the matrices
    for i in range(num):
        axs[0,i].imshow(cov_true[idx[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,i].imshow(cov_hat[idx[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)

    plt.setp(axs, xticks=[], yticks=[])

    # Label the rows
    axs[0,0].set_ylabel("True", fontsize=25)
    axs[1,0].set_ylabel("Reconstructed", fontsize=25)


    # Plot a single colorbar on the right spanning both rows 
    cbar_ax = fig.add_axes([0.93, 0.05, 0.02, 0.9])
    cbar =fig.colorbar(axs[0,0].images[0], cax=cbar_ax)

    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.01, top=0.99)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()


    return fig, axs