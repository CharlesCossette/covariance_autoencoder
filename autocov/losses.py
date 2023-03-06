import torch
from .utils import cholvec_to_covariance, tril_to_vec, vec_to_tril, tril_to_sym

def frobenius_error(
    cov1: torch.Tensor, cov2: torch.Tensor, reduction="mean", size=45, scaling = None, normalize=False
):
    """
    Frobenius norm of the error between the two matrices

    Parameters
    ----------
    cov1 : torch.Tensor
        Either batch of matrices or batch of lower triangular vectors of cholesky
    cov2 : torch.Tensor
        Either batch of matrices or batch of lower triangular vectors of cholesky
    reduction : str, optional
        by default "mean"
    size : int, optional
        size x size, of the matrix, by default 45
    """
    if cov1.shape[1] == size:
        # Then we assume that we have the full matrix
        cov1 = cov1.view(cov1.size(0), size, size)
    else:  # We assume that we have the lower triangular part
        cov1 = cholvec_to_covariance(cov1, size=size)

    if cov2.shape[1] == size:
        # Then we assume that we have the full matrix
        cov2 = cov2.view(cov2.size(0), size, size)
    else:  # We assume that we have the lower triangular part
        cov2 = cholvec_to_covariance(cov2, size=size)

    if scaling is not None:
        cov1 = cov1 * scaling
        cov2 = cov2 * scaling

    e = torch.linalg.matrix_norm(cov1 - cov2)

    if normalize:
        e = e / torch.linalg.matrix_norm(cov2)

    if reduction == "mean":
        e = torch.mean(e)
    elif reduction == "sum":
        e = torch.sum(e)
    elif reduction == "none":
        pass
    return e

def trilvec_error(trilvecs, trilvecs_out, size=45):
    trilvecs = vec_to_tril(trilvecs, 45)
    trilvecs_out = vec_to_tril(trilvecs_out, 45)

    cov = tril_to_sym(trilvecs)
    cov_out = tril_to_sym(trilvecs_out)

    e = torch.linalg.matrix_norm(
        cov - cov_out, ord="fro"
    ) / torch.linalg.matrix_norm(cov, ord="fro")
    return e


def relaxed_log_barrier(x: torch.Tensor, threshold=1e-4, relaxation=2):
    """
    Computes the relaxed log barrier function
    """
    is_below = x < threshold
    y = torch.zeros_like(x)
    y[~is_below] = -torch.log(x[~is_below])
    if relaxation == "exp":
        y[is_below] = (
            torch.exp(1 - x[is_below] / threshold)
            - 1
            - torch.log(torch.Tensor([threshold]))
        )
    elif isinstance(relaxation, int):
        k = relaxation
        y[is_below] = (k - 1) / (k) * (
            ((x[is_below] - k * threshold) / ((k - 1) * threshold)) ** k - 1
        ) - torch.log(torch.Tensor([threshold]))
    return y


def log_barrier_loss(cholvec1, cholvec2, threshold=1e-1, relaxation=2):
    """
    Computes the log barrier loss function
    """
    cov1 = cholvec_to_covariance(cholvec1)
    cov2 = cholvec_to_covariance(cholvec2)
    E = cov1 - cov2
    # J = torch.linalg.matrix_norm(E, dim=(1, 2)) - relaxed_log_barrier(
    #      torch.linalg.det(E), threshold=threshold, relaxation=relaxation
    # )
    eigs = torch.linalg.eigvalsh(E)
    is_negative = eigs <0
    eigs_scaled = torch.zeros_like(eigs)
    # eigs_scaled[is_negative] = 0.00001 * relaxed_log_barrier(eigs[is_negative])
    # eigs_scaled[~is_negative] = torch.pow(eigs[~is_negative], 2)
    J = torch.sum(torch.pow(eigs[is_negative], 2))
    
    return torch.mean(torch.linalg.matrix_norm(E)) + J
    # loss = torch.linalg.matrix_norm(E, dim=(1, 2))
    # return torch.sum(loss - torch.log(torch.linalg.det(E)))