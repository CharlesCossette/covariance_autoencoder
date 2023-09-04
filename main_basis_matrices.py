import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import traceback
import os
import autocov
from autocov.losses import trilvec_error, frobenius_error
from autocov.autoencoders import BasisModel
from autocov.utils import tril_to_sym, vec_to_tril, plot_covariances, covariance_to_cholvec, cholvec_to_covariance
# set seed 
torch.manual_seed(0)
################################################################################
dataset_kwargs = {
    "samples_per_file": -1,
    "data_dir": "/home/charles/covariance_autoencoder/data",
}
scaling = True
# model_path = "./models/autoencoder.pth"
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

num_basis_matrices = 50
num_epochs = 200
batch_size = 1000
learning_rate = 1e-3
load_saved=False
################################################################################
# Dataset and data loader
train_dataset = autocov.CovarianceDataset(train_files, **dataset_kwargs)
val_dataset = autocov.CovarianceDataset(valid_files, **dataset_kwargs)
idx = torch.linspace(0, len(train_dataset)-1, num_basis_matrices, dtype=torch.long)
idx_val = torch.linspace(0, len(val_dataset)-1, 5000, dtype=torch.long)
basis_matrices = train_dataset[idx][0]
val_cholvec = val_dataset[idx_val][0].T
# cholvec = val_cholvec


# Now, we actually seek to find the matrix M, that will minimize the reconstruction error
# I.e. e(M) = target - decode(M, encode(M, target))
# Mathematically, this is simply 
# e(M) = target - M @ w_hat 
#      = target - M @ (M.T @ M)^-1 @ M.T @ target 
#      = (I - M @ (M.T @ M)^-1 @ M.T) @ target

# M.T @ e(M) = M.T @ target - M.T @ M @ (M.T @ M)^-1 @ M.T @ target
#            = M.T @ target - M.T @ target = 0


# basis_matrices = covariance_to_cholvec(tril_to_sym(vec_to_tril(basis_matrices))) 
# Loss function and optimizer
model = BasisModel(basis_matrices.T, load_saved=load_saved)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
criterion = nn.MSELoss()
# Training loop
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
try:
    for epoch in range(1000):
        # Validation loss
        val_cholvec_hat = model(val_cholvec)

        val_loss = frobenius_error(val_cholvec_hat.T, val_cholvec.T, normalize=True)
        scheduler.step(val_loss)

        # Forward pass
        for cholvec, _ in train_loader:
            cholvec_hat = model(cholvec.T).T
            # loss = criterion(cholvec_hat, cholvec)
            loss = frobenius_error(cholvec_hat, cholvec, normalize=True)
            # loss = torch.mean(trilvec_error(cholvec, cholvec_hat ))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate error in inverses 
            cov_hat = (tril_to_sym(vec_to_tril(cholvec_hat)))
            cov = (tril_to_sym(vec_to_tril(cholvec)))
            # loss = torch.mean(torch.linalg.matrix_norm(cov_hat - cov)/torch.linalg.matrix_norm(cov))


            # Log
            print(f"Epoch: {epoch}, Train Loss: {100*loss.item():.4f}, Val Loss: {100*val_loss.item():.4f}")

except KeyboardInterrupt:
    print("Interrupted.")
except Exception as e:
    print(traceback.format_exc())
    raise

print("Training finished")
answer = input("Save model? (y/n)")
if answer == "y" or answer == "yes":
    save_path = os.path.join(os.path.dirname(__file__), "models/basis_encoder.pth")
    torch.save(model.state_dict(),save_path)
    print(f"Model saved to {save_path}")
