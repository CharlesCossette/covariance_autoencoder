from autocov import (
    IMURMIDataset,
    IMUExperimentalRMI,
    tril_to_sym,
    vec_to_tril,
    cholvec_to_covariance,
    RMICovModel,
)
import torch
from torch.utils.data import DataLoader
import os 
import traceback
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import navlie as nav
model_path = None
# model_path = "./models/imu_rmi_cov.pth"

torch.set_default_dtype(torch.float64)

encoding_size = 5
def covariance_error(trilvecs, trilvecs_out):
    trilvecs = vec_to_tril(trilvecs, 15)
    # trilvecs_out = vec_to_tril(trilvecs_out, 15)

    cov = tril_to_sym(trilvecs)
    # cov_out = tril_to_sym(trilvecs_out)

    # cov = cholvec_to_covariance(trilvecs, size=15)
    cov_out = cholvec_to_covariance(trilvecs_out, size=15)


    e = torch.linalg.matrix_norm(
        cov - cov_out, ord="fro"
    ) / torch.linalg.matrix_norm(cov, ord="fro")
    return e

import matplotlib.pyplot as plt
N_figs = 10
fig2, axes = plt.subplots(2, N_figs, figsize=(1.5*N_figs, 5))
plt.setp(axes, xticks=[], yticks=[])
img = np.zeros_like(axes)
for i in range(2):
    for j in range(N_figs):
        img[i,j] = axes[i, j].imshow(np.ones((15,15)), vmin=-10,vmax=10, cmap="RdBu")

fig2.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)


model = RMICovModel(encoding_size=encoding_size)
if model_path is not None:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

validset = IMUExperimentalRMI("./data/random2.bag")


writer = SummaryWriter(comment=f"_imu_rmi_cov_{encoding_size}")
try:
    for seed in range(80):

        data = IMURMIDataset(seed=seed, num_samples=1000, max_window_size=100)
        dataloader = DataLoader(data, batch_size=128, shuffle=True)

        # Plotting
        with torch.no_grad():
            rmis, trilvecs = validset[:]
            trilvecs_out = model(rmis, trilvecs)
            cov = tril_to_sym(vec_to_tril(trilvecs, size=15))

            cov_out = cholvec_to_covariance(trilvecs_out, size=15)

            e = torch.linalg.matrix_norm(
                cov - cov_out, ord="fro"
            ) / torch.linalg.matrix_norm(cov, ord="fro")
            loss = torch.mean(e)
            idx_j = np.linspace(0, len(validset)-1, N_figs).astype(int)
            val_cov_plot = cov[idx_j]
            val_output_plot = cov_out[idx_j]
            for j in range(N_figs):
                img[0,j].set_data(val_cov_plot[j].detach().numpy())
                img[1,j].set_data(val_output_plot[j].detach().numpy())
                axes[0,j].set_title(f"t={idx_j[j]}")

        scheduler.step(loss)
        writer.add_scalar("Loss", loss.item(), seed)
        writer.add_scalar("MeanError", torch.mean(e).item(), seed)
        writer.add_scalar("MaxError", torch.max(e).item(), seed)
        writer.add_figure("Covariance", fig2, seed)

        # Training
        for epoch in range(50):
            for rmis, trilvecs in dataloader:
                optimizer.zero_grad()
                trilvecs_out = model(rmis, trilvecs)
                e = covariance_error(trilvecs, trilvecs_out)
                loss = torch.mean(e)
                loss.backward()
                optimizer.step()
                e = torch.mean(e)
                print(
                    f"Seed: {seed} Epoch: {epoch}, Error: {e.item()*100:0.4f}"
                )


except KeyboardInterrupt:
    print("Interrupted.")
except Exception as e:
    print(traceback.format_exc())
    raise

print("Training finished")
answer = input("Save model? (y/n)")
if answer == "y" or answer=="yes":
    torch.save(model.state_dict(), "./models/imu_rmi_cov.pth")
    print("Model saved to ./models/imu_rmi_cov.pth")
