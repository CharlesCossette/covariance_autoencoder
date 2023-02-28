import torch
import pynav as nav
from pylie import SE23, SO3
import multinav.arpimu
import multinav.mrfilter
import numpy as np
from .utils import tril_to_vec, covariance_to_cholvec
import pickle
import os 
from torch.utils.data import Dataset

def _load_covariance_file(results_file):

    # Check cache first
    if os.path.exists("./cache/" + results_file + ".cache"):
        return torch.load("./cache/" + results_file + ".cache")
    else:
        results = pickle.load(open("./results/" + results_file, "rb"))
        _identity = multinav.arpimu.ARPIMUState.from_absolute_states(
            [
                nav.lib.IMUState(
                    SE23.identity(), [0, 0, 0], [0, 0, 0], state_id=i
                )
                for i in range(3)
            ],
            1,
        )

        # assumes results is a dict
        _covariances = []
        _states = []
        for agent_name, agent_results in results.items():
            agent_results: nav.GaussianResultList = agent_results

            _covariances.append(agent_results.covariance)
            _states.extend(
                [
                    x.minus(_identity).ravel()
                    for x in agent_results.state
                ]
            )

        _covariances = torch.tensor(np.concatenate(_covariances, axis=0))
        _states = torch.tensor(_states)


        # Cache results
        torch.save(
            (_covariances, _states), "./cache/" + results_file + ".cache"
        )
        return _covariances, _states


def _load_covariance_data(results_file_list, clip_end=True):

    # Check if list or single file was passed
    if isinstance(results_file_list, list) or isinstance(
        results_file_list, tuple
    ):
        pass
    else:
        results_file_list = [results_file_list]

    _covariances = None
    _states = None
    for results_file in results_file_list:

        _cov, _s = _load_covariance_file(results_file)

        if clip_end:
            if _cov.shape[0] > 5000:
                _cov = _cov[:5000]
                _s = _s[:5000]

        if _covariances is None:
            _covariances = _cov
        else:
            _covariances = torch.cat((_covariances, _cov))

        if _states is None:
            _states = _s
        else:
            _states = torch.cat((_states, _s))

    return _covariances, _states


class CovarianceDataset(Dataset):
    unit_scaling = [
        0.01,
        0.01,
        0.01,
        0.2,
        0.2,
        0.2,
        0.4,
        0.4,
        0.4,
        0.001,
        0.001,
        0.001,
        0.1,
        0.1,
        0.1,
    ] * 3

    _s = torch.Tensor(unit_scaling).reshape((-1, 1))
    #scaling_matrix =  _s @ _s.T 
    scaling_matrix = torch.ones((45,45))
    def __init__(self, results_file_list, clip_end=False):

        self._covariances, self._states = _load_covariance_data(
            results_file_list, clip_end=clip_end
        )
        
        self._covariances = torch.true_divide(self._covariances, self.scaling_matrix)

        self._covariances = covariance_to_cholvec(self._covariances)

    def __len__(self):
        return len(self._covariances)

    def __getitem__(self, idx):
        cov = self._covariances[idx]
        dx = self._states[idx]
        return cov, dx






class IMURMIDataset(Dataset):
    def __init__(
        self, seed=0, num_samples=1000, mean_freq=200, max_window_size=10
    ):

        cache_file = (
            f"imu_rmi_{seed}_{num_samples}_{mean_freq}_{max_window_size}"
        )
        if os.path.exists("./cache/" + cache_file + ".cache"):
            temp = torch.load("./cache/" + cache_file + ".cache")
            self._rmi_data = temp["rmi_data"]
            self._cov_data = temp["cov_data"]
        else:
            # TODO: vary Q
            # TODO: vary biases
            self._seed = seed
            self._num_samples = num_samples

            Q = np.diag(
                [
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.00174**2,
                    0.00174**2,
                    0.00174**2,
                    0.01**2,
                    0.01**2,
                    0.01**2,
                ]
            )  # Todo. parameterize
            dt_mean = 1 / mean_freq
            rmi_data = []
            cov_data = []
            for i in range(num_samples):
                window_size = (
                    int(np.random.uniform(0, max_window_size)) + 1
                )  # todo: vectorize
                rmi = nav.lib.IMUIncrement(Q, [0, 0, 0], [0, 0, 0], window_size)

                gyro_data = np.random.normal(0, 1, (window_size, 3))
                accel_data = np.random.normal(0, 1, (window_size, 3))
                C_ba = SO3.Exp(np.random.normal(0, 0.4, 3))

                g_a = np.array([0, 0, -9.80665])
                accel_data = accel_data - C_ba @ g_a

                dt_std_dev = 0.1 * dt_mean
                dts = np.random.normal(1 / mean_freq, dt_std_dev, window_size)
                stamps = np.cumsum(dts)
                for j in range(window_size):
                    imu = nav.lib.IMU(gyro_data[j], accel_data[j], stamps[j])
                    rmi.increment(imu, dts[j])

                temp = np.concatenate(
                    [
                        SO3.Log(rmi.value[0:3, 0:3]).ravel(),
                        rmi.value[0:3, 3],
                        rmi.value[0:4, 4],
                    ]
                )
                rmi_data.append(temp)
                cov_data.append(rmi.covariance)

            self._rmi_data = torch.Tensor(np.array(rmi_data))
            self._cov_data = torch.Tensor(np.array(cov_data))

            # lower triangular entries
            self._cov_data = tril_to_vec(self._cov_data) / 1e-8
            torch.save(
                {
                    "rmi_data": self._rmi_data,
                    "cov_data": self._cov_data,
                },
                "./cache/" + cache_file + ".cache",
            )

    def __len__(self):
        return self._rmi_data.shape[0]

    def __getitem__(self, idx):
        return self._rmi_data[idx], self._cov_data[idx]


from pymocap import IMUData


class IMUExperimentalRMI(Dataset):
    def __init__(self, bagfile: str):

        # Get only filename
        bagname = bagfile.split("/")[-1]
        bagname = bagname.split(".")[0]

        cache_file = f"imu_rmi_{bagname}"
        if os.path.exists("./cache/" + cache_file + ".cache"):
            temp = torch.load("./cache/" + cache_file + ".cache")
            self._rmi_data = temp["rmi_data"]
            self._cov_data = temp["cov_data"]
        else:

            Q = np.diag(
                [
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.03**2,
                    0.00174**2,
                    0.00174**2,
                    0.00174**2,
                    0.01**2,
                    0.01**2,
                    0.01**2,
                ]
            )  # Todo. parameterize

            rmi_data = []
            cov_data = []

            for agent in ["ifo001", "ifo002", "ifo003"]:
                imu_data = IMUData.from_bag(
                    bagfile, f"/{agent}/mavros/imu/data_raw"
                )
                imu_list = imu_data.to_pynav()

                # Divide into chunks of 10
                for i in range(0, len(imu_list), 11):
                    imu_chunk = imu_list[i : i + 10]
                    rmi = nav.lib.IMUIncrement(Q, [0, 0, 0], [0, 0, 0])
                    for j in range(1, len(imu_chunk)):
                        imu = imu_chunk[j]
                        rmi.increment(imu, imu.stamp - imu_chunk[j - 1].stamp)

                    temp = np.concatenate(
                        [
                            SO3.Log(rmi.value[0:3, 0:3]).ravel(),
                            rmi.value[0:3, 3],
                            rmi.value[0:4, 4],
                        ]
                    )
                    rmi_data.append(temp)
                    cov_data.append(rmi.covariance)

            self._rmi_data = torch.Tensor(np.array(rmi_data))
            self._cov_data = torch.Tensor(np.array(cov_data))

            # lower triangular entries
            self._cov_data = tril_to_vec(self._cov_data) / 1e-8
            torch.save(
                {
                    "rmi_data": self._rmi_data,
                    "cov_data": self._cov_data,
                },
                "./cache/" + cache_file + ".cache",
            )

    def __len__(self):
        return self._rmi_data.shape[0]

    def __getitem__(self, idx):
        return self._rmi_data[idx], self._cov_data[idx]