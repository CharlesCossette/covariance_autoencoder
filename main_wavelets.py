# %%
import pickle
import navlie as nav
import numpy as np
import matplotlib.pyplot as plt
import torch
import pywt
from typing import Dict
from scipy.sparse import coo_matrix
from scipy.linalg import null_space
# from multinav.autoencoders import Autoencoder, CovarianceDataset, frobenius_error, cholvec_to_covariance

class FFTCompressor:
    def __init__(self, retain_ratio=0.75, chol=False, dim=None):
        self.retain_ratio = retain_ratio
        self.dim = dim
        self.chol = chol

    def encode(self, covariance):
        if self.dim is None:
            self.dim = covariance.shape[0]

        if self.chol:
            cov_chol = np.linalg.cholesky(covariance)
        else:
            cov_chol = covariance

        scale = np.max(np.abs(cov_chol))
        cov_chol = cov_chol / scale
        cov_chol_fourier = np.fft.fft2(cov_chol)
        fourier_sorted = np.sort(np.abs(cov_chol_fourier.ravel()))
        thresh = fourier_sorted[
            int(np.floor((1 - self.retain_ratio) * fourier_sorted.size))
        ]
        retain_mask = np.abs(cov_chol_fourier) > thresh
        cov_chol_fourier[~retain_mask] = 0
        cov_chol_fourier = coo_matrix(cov_chol_fourier)
        return cov_chol_fourier, scale

    def get_encoding_size(self, cov_chol_fourier, scale):
        return cov_chol_fourier.nnz * (8 + 1) + 8

    def decode(self, cov_chol_fourier, scale):
        cov_chol_fourier = cov_chol_fourier.toarray()
        cov_chol = np.fft.ifft2(cov_chol_fourier).real
        cov_chol = cov_chol * scale
        if self.chol:
            cov = cov_chol @ cov_chol.T
        else:
            cov = cov_chol
        return cov

    def __repr__(self):
        return f"FFTCompressor(retain_ratio={self.retain_ratio})"


class WaveletCompressor:
    def __init__(
        self, retain_ratio=0.1, wavelet="db1", levels=4, chol=False, dim=None
    ):
        self.retain_ratio = retain_ratio
        self.dim = dim
        self.chol = chol
        self.wavelet = wavelet
        self.levels = levels

    def encode(self, covariance):
        if self.dim is None:
            self.dim = covariance.shape[0]

        if self.chol:
            cov_chol = np.linalg.cholesky(covariance)
        else:
            cov_chol = covariance

        scale = np.max(np.abs(cov_chol))
        cov_chol = cov_chol / scale
        n = self.levels
        w = self.wavelet
        coeffs = pywt.wavedec2(cov_chol, wavelet=w, level=n)

        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        Csort = np.sort(np.abs(coeff_arr.reshape(-1)))
        keep = self.retain_ratio
        thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr * ind  # Threshold small indices
        return (coo_matrix(Cfilt), coeff_slices, scale)

    def get_encoding_size(self, Cfilt, coeff_slices, scale):
        sum = 0
        for slc in coeff_slices[0]:
            sum += 2
        for temp in coeff_slices[1:]:
            for slc in temp.values():
                sum += 4
        return (Cfilt.nnz) * (8 + 1) + 8 + sum

    def decode(self, Cfilt, coeff_slices, scale):
        Cfilt = Cfilt.toarray()
        coeff_slices = coeff_slices
        coeffs_filt = pywt.array_to_coeffs(
            Cfilt, coeff_slices, output_format="wavedec2"
        )

        cov_chol = pywt.waverec2(coeffs_filt, wavelet=self.wavelet)
        cov_chol = cov_chol[: self.dim, : self.dim]
        cov_chol = cov_chol * scale

        if self.chol:
            cov = cov_chol @ cov_chol.T
        else:
            cov = cov_chol
        return cov

    def __repr__(self):
        return f"WaveletCompressor(retain_ratio={self.retain_ratio})"


class EigenCompressor:
    def __init__(self, num_vecs=10):
        self.num_vecs = num_vecs

    def encode(self, covariance):
        L, V = np.linalg.eigh(covariance)
        idx = np.argsort(L)
        L = L[idx]
        idx_small = idx[: self.num_vecs]
        idx_large = idx[-self.num_vecs :]
        V_small = V[:, 0].reshape(V.shape[0], -1)
        V_large = V[:, idx_large]
        L = L
        return L, V_small, V_large

    def get_encoding_size(self, L, V_small, V_large):
        return (L.size + V_small.size + V_large.size) * 8

    def decode(self, L, V_small, V_large):
        V = np.hstack((V_small, V_large))
        N = null_space(V.T)
        V = np.hstack((V_small, N, V_large))
        cov = V @ np.diag(L) @ V.T
        return cov

    def __repr__(self):
        return f"EigenCompressor(num_vecs={self.num_vecs})"


def test_compressor(compressor: FFTCompressor, cov):
    out = compressor.encode(cov)
    cov2 = compressor.decode(*out)
    num = compressor.get_encoding_size(*out)

    min_ = min(cov.min(), cov2.min())
    max_ = max(cov.max(), cov2.max())

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    im = ax[0].imshow(cov, vmin=min_, vmax=max_)
    im2 = ax[1].imshow(cov2, vmin=min_, vmax=max_)
    im3 = ax[2].imshow(np.abs(cov - cov2))
    ax[0].set_title(f"Original \n {cov.shape[0]**2*8} bytes")
    ax[1].set_title(f"Compressed\n {num} bytes")
    ax[2].set_title(f"Error")
    e = covariance_error(cov, cov2)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    fig.colorbar(im2, ax=ax[1], fraction=0.046)
    fig.colorbar(im3, ax=ax[2], fraction=0.046)
    plt.suptitle(f"{compressor.__repr__()} Error: {e*100:.2f}%")

    return fig, ax


def covariance_error(cov1, cov2):
    cov1 = cov1.flatten()
    cov2 = cov2.flatten()
    return np.linalg.norm(cov1 - cov2) / np.linalg.norm(cov1)


def compare_compressors(compressor_dict, cov):
    fig, ax = plt.subplots(1, 1)
    for name, compressors in compressor_dict.items():
        sz = []
        e = []
        for compressor in compressors:
            out = compressor.encode(cov)
            cov2 = compressor.decode(*out)
            sz.append(compressor.get_encoding_size(*out))
            e.append(covariance_error(cov, cov2))
        sz = np.array(sz)
        e = np.array(e)
        ax.plot(sz, e, "s-", label=f"{name}")

    raw_size = cov.shape[0] ** 2 * 8
    ax.plot([raw_size, raw_size], [0, 1], "--", label="Raw")
    chol_size = sum(range(cov.shape[0] + 1)) * 8
    ax.plot([chol_size, chol_size], [0, 1], ":", label="Simple cholesky")
    ax.set_xlabel("Encoding Size (bytes)")
    ax.set_ylabel("Error")
    ax.legend()
    return fig, ax


#%%
test_idx = 4000
test_file = "res_random2_decent_15_55.p"
dataset = CovarianceDataset(test_file, augment_diagonal=False, augment_first_chunk=False)
cov, x = dataset[test_idx]
# %%
import seaborn as sns

sns.set_theme(style="whitegrid")

# compressor = FFTCompressor(retain_ratio=0.2, chol=False)
# compressor = WaveletCompressor(retain_ratio=0.2, chol=False)
compressor = EigenCompressor(num_vecs=6)
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("./models/autoencoder.pth"))
cov_mat = cholvec_to_covariance(cov)[0]
test_compressor(compressor, cov_mat.numpy())
plt.tight_layout()

compressor_dict = {
    "FFT": [
        FFTCompressor(retain_ratio=r) for r in np.linspace(0.01, 1, 20)
    ],
    "FFT on chol": [
        FFTCompressor(retain_ratio=r, chol=True)
        for r in np.linspace(0.01, 1, 20)
    ],
    "Wavelet": [
        WaveletCompressor(retain_ratio=r) for r in np.linspace(0.01, 1, 20)
    ],
    "Wavelet on chol": [
        WaveletCompressor(retain_ratio=r, chol=True)
        for r in np.linspace(0.01, 1, 20)
    ],
    "Eigen": [EigenCompressor(num_vecs=n) for n in range(1, 44)],
}
fig, ax = compare_compressors(compressor_dict, cov_mat.numpy())

#%% Test autoencoder
cov, x = dataset[:]
cov_out = autoencoder((cov,x))
err = frobenius_error(cov, cov_out, reduction="none")

# Plot
ax.scatter(
    45 * 8,
    err[test_idx].detach().numpy(),
    s=200,
    marker="*",
    color=(1, 0, 0),
    label="AutoEncoder",
)
ax.legend()
fig, ax = plt.subplots(1, 1)
ax.scatter(range(len(dataset)), err.detach().numpy())


cov = cholvec_to_covariance(cov).detach().numpy()[test_idx]
cov_out = cholvec_to_covariance(cov_out).detach().numpy()[test_idx]
fig, ax = plt.subplots(1, 3, figsize=(8, 4))
min_ = min(cov.min(), cov_out.min())
max_ = max(cov.max(), cov_out.max())
im = ax[0].imshow(cov, vmin=min_, vmax=max_)
im2 = ax[1].imshow(cov_out, vmin=min_, vmax=max_)
im3 = ax[2].imshow(np.abs(cov - cov_out))
ax[0].set_title(f"Original \n {cov.shape[0]**2*8} bytes")
ax[1].set_title(f"Compressed\n {45*8} bytes")
ax[2].set_title(f"Error")
fig.suptitle("AutoEncoder")
plt.show()
# %%

# %%
