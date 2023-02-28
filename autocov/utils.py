import torch

def tril_to_vec(x: torch.Tensor, size=None):
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