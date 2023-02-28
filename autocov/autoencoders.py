import torch
import torch.nn as nn
import pynav as nav
from pylie import SE23, SO3
import multinav.arpimu
import multinav.mrfilter
import numpy as np
from .utils import tril_to_vec, tril_to_sym, vec_to_tril, cholvec_to_covariance
from .datasets import CovarianceDataset, arpimu_identity_state

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1035, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 90),
        )
        self.decoder = nn.Sequential(
            nn.Linear(90, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 1035),
        )
        #self.residual_matrix = nn.Parameter(torch.randn((1035-45, 90)))

    def forward_from_pynav(self, x: multinav.mrfilter.MRFilterState):
        """
        Parameters
        ----------
        x : MRFilterState
            state to encode
        """
        with torch.no_grad():
            cov_in = torch.Tensor(x.covariance).unsqueeze(0)
            cov_in = torch.true_divide(cov_in, CovarianceDataset.scaling_matrix)
            dx = torch.Tensor(x.state.minus(arpimu_identity_state).ravel()).unsqueeze(
                0
            )
            chol_in = tril_to_vec(torch.linalg.cholesky(cov_in))
            chol_vec_out = self.forward((chol_in, dx))
            cov_out = cholvec_to_covariance(chol_vec_out, 45).squeeze(0)
            cov_out = cov_out * CovarianceDataset.scaling_matrix
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
        chol_in, dx_in = x_in
        # scale = torch.max(torch.abs(chol_in), dim=1, keepdim=True)[0]
        # chol_in = chol_in / scale

        # Autoencode
        cov_mat = vec_to_tril(chol_in, 45)
        diagonals = torch.diagonal(cov_mat, dim1=-2, dim2=-1)

        x = chol_in
        x = self.encoder(x)


        # Decode
        x = torch.cat((x, diagonals), dim=1)

        # Decoder with skip connection
        chol_out = self.decoder(x) #+ torch.matmul(x, self.residual_matrix.T)
        

        chol_out = vec_to_tril(chol_out, 45, offset=-1)  # Make back into triangular matrix
        chol_out = chol_out + torch.diag_embed(
            diagonals
        )  # Add diagonals (skip connection)
        chol_out = tril_to_vec(chol_out, 45)
        # chol_out = chol_out * scale
        return chol_out


################################################################################
################################################################################
################################################################################


class RMICovModel(torch.nn.Module):
    def __init__(self):
        super(RMICovModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(120, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(12, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 120),
        )
        self.residual_matrix = nn.Parameter(torch.randn((120, 12)))

    def forward_from_pynav(self, rmi: nav.lib.RelativeMotionIncrement):
        cov = torch.from_numpy(rmi.covariance)
        cov = cov.unsqueeze(0)
        trilvecs = tril_to_vec(cov, 15) / 1e-8
        dt = rmi.value[3,4]
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
        enc = self.encoder(trilvecs)
        x = torch.cat((x, enc), dim=1)
        x = self.decoder(x) + torch.matmul(x, self.residual_matrix.T)
        # x = x*torch.pow(dts, 2).unsqueeze(1)
        return x
