from autocov import (
    IMURMIDataset,
    IMUExperimentalRMI,
    tril_to_sym,
    vec_to_tril,
    RMICovModel,
)
import torch
import matplotlib.pyplot as plt
from autocov.utils import plot_covariances

torch.set_default_dtype(torch.float64)
validset = IMUExperimentalRMI("./data/random2.bag")
plt.rcParams["text.usetex"] = True


from tensorboard.backend.event_processing import event_accumulator
import os
import seaborn as sns
# Get all the event files in the runs directory, including subdirectories
runs_error_data = {}
for root, dirs, files in os.walk("paper"):
    # Only if folder name contains "imu_rmi_cov"
    if "imu_rmi_cov" in root:
        # Get encoding size, embedded in the name immediately after imu_rmi_cov
        encoding_size = int(root.split("imu_rmi_cov")[1].split("_")[1])

        for file in files:
            if file.startswith("events.out.tfevents"):
                filename = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(filename)
                ea.Reload()
                data_x = torch.Tensor([s.step for s in ea.Scalars("MeanError")])
                data_y = torch.Tensor([s.value for s in ea.Scalars("MeanError")])
                runs_error_data[encoding_size] = torch.vstack((data_x, data_y))

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1,1, figsize=(8,3))
data = runs_error_data[0]
ax.plot(data[0], data[1]*100, label="No encoding", color="r")
for encoding_size, data in runs_error_data.items():
    if encoding_size != 0:
        ax.plot(data[0], data[1]*100, label=f"Encoding size {encoding_size}", color="b", alpha = 1-encoding_size/15)
ax.set_xlabel("Thousands of samples seen", fontsize=18)
ax.set_ylabel("Mean error (\%)", fontsize=18)
ax.set_ylim(0, 25)
ax.legend()
ax.set_title("Validation error during training", fontsize=18)
plt.tight_layout()
model = RMICovModel(load_saved=True)
unit_scaling = [
    0.0514,
    0.0514,
    0.0514,
    0.0265,
    0.0265,
    0.0257,
    0.0034,
    0.0034,
    0.0033,
    0.01,
    0.01,
    0.01,
    0.03,
    0.03,
    0.03,
]
_s = torch.Tensor(unit_scaling).reshape((-1, 1))
scaling_matrix = _s @ _s.T
# scaling_matrix = torch.ones_like(scaling_matrix)


# Plotting
with torch.no_grad():
    rmis, trilvecs = validset[:]
    trilvecs_out = model(rmis, trilvecs)
    trilvecs = vec_to_tril(trilvecs, 15)
    trilvecs_out = vec_to_tril(trilvecs_out, 15)

    cov = 1e-8 * tril_to_sym(trilvecs) / scaling_matrix
    cov_out = 1e-8 * tril_to_sym(trilvecs_out) / scaling_matrix
    fig, ax = plot_covariances(cov, cov_out, num=3, cmap="RdBu", vmax=0.0001)

plt.show()
