import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import traceback
import numpy as np
import autocov

################################################################################
# Hyperparameters
num_epochs = 200
batch_size = 256
learning_rate = 1e-3
dataset_kwargs = {
    "samples_per_file": -1,
    "data_dir": "/home/charles/covariance_autoencoder/data",
}
scaling = True
load_saved=False
train_file = [
    # "res_bias_calib2_decent_15_200_high_freq.p",
    "res_simple_los_2022_08_03_12_41_20_decent_15_40.p",
    "res_random1_decent_10_100.p",
    "res_bias_calib2_decent_15_200_lower_freq_pre_fusion.p",
    "results_imu_decent.p"
]
valid_file = [
    "res_random2_decent_15_55_lower_freq.p",
    "res_random2_decent_15_55.p",
]
model_path = None
# model_path = "./models/autoencoder.pth"
# Get list of files in data directory
train_files = os.listdir(dataset_kwargs["data_dir"])

valid_files = ["res_random2_decent_15_55.p"]

exclude_files = [
    # "res_bias_calib2_decent_15_200_high_freq.p",
    # "res_bias_calib2_decent_15_200_lower_freq_pre_fusion.p",
]
# Remove validation files from training files
for file in valid_files + exclude_files:
    train_files.remove(file)
# train_file = valid_file

################################################################################
# Dataset and data loader
train_dataset = autocov.CovarianceDataset(train_files, **dataset_kwargs)
val_dataset = autocov.CovarianceDataset(valid_files, **dataset_kwargs)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
)


# Loss function and optimizer
model = autocov.Autoencoder(load_saved=load_saved)
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Tensorboard
writer = SummaryWriter()
val_dataset_len = 2000
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
val_plot = ax.plot(range(val_dataset_len), range(val_dataset_len))[0]
ax.set_ylim(0, 1)
ax.set_xlabel("Time step")
ax.set_ylabel("Frobenius error")
ax.set_title("Reconstruction error on validation trajectory")


N_figs = 10
fig2, axes = plt.subplots(2, N_figs, figsize=(1.5 * N_figs, 5))
plt.setp(axes, xticks=[], yticks=[])
cholvec = train_dataset[2000][0]
cov = (
    autocov.cholvec_to_covariance(cholvec.unsqueeze(0), 45).detach().numpy()[0]
)
img = np.zeros_like(axes)
for i in range(2):
    for j in range(N_figs):
        img[i, j] = axes[i, j].imshow(
            cov, cmap="RdBu_r"
        )

fig2.subplots_adjust(
    wspace=0.0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0
)


N_eigs = 300
fig3, ax = plt.subplots(1, 1, figsize=(10, 5))
im_eig = ax.imshow(
    np.zeros((45, N_eigs)), cmap="seismic_r", vmin=-3, vmax=3
)
ax.set_ylabel("Eigenvalue of cov_hat - cov")
ax.set_xlabel("Data Sample #")
ax.set_title("Eigenvalues of Error")
fig3.colorbar(im_eig, ax=ax, fraction=0.046, pad=0.04)


# Training loop
try:
    for epoch in range(num_epochs):
        # Evaluate the model on the validation set
        with torch.no_grad():
            val_cholvec, val_x = val_dataset[:val_dataset_len]
            val_cholvec_output = model((val_cholvec, val_x))
            val_cholvec = model.normalize(val_cholvec)
            val_cholvec_output = model.normalize(val_cholvec_output)
            val_error = autocov.frobenius_error(
                val_cholvec_output,
                val_cholvec,
                reduction="none",
                normalize=True,
            )
            mean_val_error = torch.mean(val_error)
            val_plot.set_ydata(val_error)
            val_cov = autocov.cholvec_to_covariance(val_cholvec, 45)
            val_cov_output = autocov.cholvec_to_covariance(
                val_cholvec_output, 45
            )
            idx_j = np.linspace(0, val_dataset_len - 1, N_figs).astype(int)
            E = val_cov_output - val_cov
            eigs = torch.linalg.eigvalsh(E).T
            eigs = eigs[
                :, np.linspace(0, val_dataset_len - 1, N_eigs).astype(int)
            ]
            eigs = eigs.detach().numpy()
            eigs = np.flipud(eigs)
            im_eig.set_data(eigs)
            val_cov_plot = val_cov[idx_j]
            val_output_plot = val_cov_output[idx_j]
            for j in range(N_figs):
                img[0, j].set_data(val_cov_plot[j].detach().numpy())
                img[1, j].set_data(val_output_plot[j].detach().numpy())
                axes[0, j].set_title(f"t={idx_j[j]}")

        # Train the model
        for cholvec, x in train_loader:
            # Forward pass
            output = model((cholvec, x))
            cholvec = model.normalize(cholvec)
            output = model.normalize(output)
            # loss = criterion(cholvec, output)
            loss =  autocov.frobenius_error(cholvec, output, normalize=True)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e = autocov.frobenius_error(output, cholvec, normalize=True)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                + f"Loss: {loss.item():.4e}, "
                + f"Error: {e.item()*100:.4f}, "
                + f"Val Error: {mean_val_error.item()*100:.4f}"
            )

        # Write to tensorboard
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalars(
            "Error", {"train": e.item(), "val": mean_val_error.item()}, epoch
        )
        writer.add_figure("Reconstruction Error", fig, epoch)
        writer.add_figure("Covariance", fig2, epoch)
        writer.add_figure("Eigenvalues", fig3, epoch)

except KeyboardInterrupt:
    print("Interrupted.")
except Exception as e:
    print(traceback.format_exc())
    raise

print("Training finished")
answer = input("Save model? (y/n)")
if answer == "y" or answer == "yes":
    torch.save(model.state_dict(), "./models/autoencoder.pth")
    print("Model saved to ./models/autoencoder.pth")
