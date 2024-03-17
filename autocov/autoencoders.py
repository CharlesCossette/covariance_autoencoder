import torch
import torch.nn as nn
import navlie as nav
from pymlg import SE23, SO3
import multinav.arpimu
import multinav.mrfilter
import numpy as np
from .utils import tril_to_vec, tril_to_sym, vec_to_tril, cholvec_to_covariance, covariance_to_cholvec
import autocov.datasets as datasets
import os

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, normalize=False, load_saved=False):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1035, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(128, 15),
        )
        self.decoder = nn.Sequential(
            nn.Linear(45+15, 128),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1035),
        )
        # self.residual_matrix = nn.Parameter(torch.randn((1035-45, 90)))

        if normalize:
            normalization = torch.load("./models/normalization.pth")
            self.mean = normalization["mean"]
            self.std = normalization["std"]
        else:
            self.mean = torch.zeros(1080)
            self.std = torch.ones(1080)
        self.do_normalization = normalize

        if load_saved:
            # model file relative to this file
            model_file = os.path.join(
                os.path.dirname(__file__), "../models/autoencoder.pth"
            )
            self.load_state_dict(torch.load(model_file))

    def normalize(self, x):
        if not self.do_normalization:
            return x

        return x



    def denormalize(self, cholvec):
        if not self.do_normalization:
            return cholvec
        return cholvec

    def forward_from_navlie(self, x: multinav.mrfilter.MRFilterState):
        """
        Parameters
        ----------
        x : MRFilterState
            state to encode
        """
        with torch.no_grad():
            cov_in = torch.Tensor(x.covariance).unsqueeze(0)
            dx = torch.Tensor(
                x.state.minus(datasets.arpimu_identity_state).ravel()
            ).unsqueeze(0)
            chol_in = tril_to_vec(torch.linalg.cholesky(cov_in))
            chol_vec_out = self.forward((chol_in, dx))
            cov_out = cholvec_to_covariance(chol_vec_out, 45).squeeze(0)
            cov_out = (cov_out + cov_out.T) / 2
            cov_out = cov_out.detach().numpy()

        return cov_out

    def forward(self, x_in):
        """
        Parameters
        ----------
        x_in : Tuple[torch.Tensor, torch.Tensor]
            first element is the flattened cholesky decomp of covariance
            with shape (batch_size, 1035)
            second element is the state difference with shape (batch_size, 45)
        """
        # chol_in, dx_in = x_in
        # scale = torch.max(torch.abs(chol_in), dim=1, keepdim=True)[0]
        chol_in, dx_in = self.normalize(x_in)

        # Autoencode
        # cov_mat = vec_to_tril(chol_in, 45)
        # diagonals = torch.diagonal(cov_mat, dim1=-2, dim2=-1)

        x = chol_in
        x = self.encoder(x)

        # Decode
        x = torch.cat((x, dx_in), dim=1)

        # Decoder with skip connection
        chol_out = self.decoder(x)  # + torch.matmul(x, self.residual_matrix.T)

        # chol_out = vec_to_tril(chol_out, 45)  # Make back into triangular matrix
        # chol_out = chol_out + torch.diag_embed(
        #     diagonals
        # )  # Add diagonals (skip connection)
        # chol_out = tril_to_vec(chol_out, 45)

        chol_out = self.denormalize(chol_out)
        return chol_out


################################################################################
################################################################################
################################################################################


class RMICovModel(torch.nn.Module):
    def __init__(self, load_saved=False, encoding_size=2):
        super(RMICovModel, self).__init__()
        self.encoding_size = encoding_size
        self.encoder = nn.Sequential(
            nn.Linear(120, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, encoding_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10 + encoding_size, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 120),
        )
        #self.residual_matrix = nn.Parameter(torch.randn((120, 10 + encoding_size)))

        if load_saved:
            # model file relative to this file
            model_file = os.path.join(
                os.path.dirname(__file__), "../models/imu_rmi_cov.pth"
            )
            self.load_state_dict(torch.load(model_file))

    def forward_from_navlie(self, rmi: nav.lib.RelativeMotionIncrement):
        cov = torch.from_numpy(rmi.covariance)
        cov = cov.unsqueeze(0)
        trilvecs = tril_to_vec(cov, 15) / 1e-8
        dt = rmi.value[3, 4]
        if dt == 0:
            cov_out = rmi.covariance
        else:
            x = torch.from_numpy(
                np.concatenate(
                    [
                        SO3.Log(rmi.value[0:3, 0:3]).ravel(),
                        rmi.value[0:3, 3],
                        rmi.value[0:4, 4],
                    ]
                )
            ).unsqueeze(0)
            trilvecs_out = self.forward(x, trilvecs)
            cov_out = vec_to_tril(trilvecs_out, 15) * 1e-8
            cov_out = tril_to_sym(cov_out)
        return cov_out.squeeze(0).detach().numpy()

    def forward(self, x, trilvecs):
        if self.encoding_size > 0:
            enc = self.encoder(trilvecs)
            x = torch.cat((x, enc), dim=1)
        x = self.decoder(x)# + torch.matmul(x, self.residual_matrix.T)

        # x is now a [batch x 120] vector representing the lower triangle of a cholesky decomposition.
        # To ensure positive definiteness, we square the diagonal elements.
        x = vec_to_tril(x, 15) # x is now [batch x 15 x 15]

        # Get diagonal ements
        diag = torch.diagonal(x, dim1=-2, dim2=-1)

        # Remove diagonal from the 15x15 matrices and re-add the squared diagonal
        x = x - torch.diag_embed(diag)
        x = x + torch.diag_embed(diag ** 2)
        x = tril_to_vec(x)

        return x


################################################################################
################################################################################
################################################################################

class BasisModel(nn.Module):
    def __init__(self, basis_matrix=None, load_saved=False):
        super().__init__()
        self.basis_matrix = nn.Parameter(torch.zeros((1035, 200)))
        if basis_matrix is not None:
            self.basis_matrix = nn.Parameter(basis_matrix.clone().contiguous())

        if load_saved:
            # model file relative to this file
            model_file = os.path.join(
                os.path.dirname(__file__), "../models/basis_encoder.pth"
            )
            self.load_state_dict(torch.load(model_file))


        self.static_matrices = tril_to_vec(torch.eye(45).unsqueeze(0)).reshape((-1, 1))


    def forward_from_navlie(self, x: multinav.mrfilter.MRFilterState):
        """
        Parameters
        ----------
        x : MRFilterState
            state to encode
        """
        with torch.no_grad():
            cov_in = torch.Tensor(x.covariance).unsqueeze(0) 
            cov_in = cov_in / datasets.CovarianceDataset.scaling_matrix
            # cov_in = torch.linalg.inv(cov_in)
            cholvec_in = covariance_to_cholvec(cov_in + 1e-4*torch.eye(45))
            cholvec_out = self.forward(cholvec_in.T).T
            cov_out = cholvec_to_covariance(cholvec_out).squeeze(0)
            # cov_out = tril_to_sym(vec_to_tril(cholvec_out)).squeeze(0)
            # cov_out = torch.linalg.inv(cov_out)
            cov_out = cov_out * datasets.CovarianceDataset.scaling_matrix
            cov_out = (cov_out + cov_out.T) / 2
            cov_out = cov_out.detach().numpy()

        return cov_out

    def encode(self, target):
        M = torch.hstack((self.basis_matrix, self.static_matrices))
        w_hat = torch.linalg.solve(M.T @ M, M.T @ target)
        return w_hat

    def decode(self, w_hat):
        """

        Parameters
        ----------
        w_hat : [45 x N]
        Returns
        -------
        target_hat : [1035 x N]
        """
        M = torch.hstack((self.basis_matrix, self.static_matrices))
        cholvec_hat = M @ w_hat
        return cholvec_hat #[1035 x N]
    
    def forward(self, target):
        w_hat = self.encode(target)
        cholvec_hat = self.decode(w_hat)
        return cholvec_hat