import torch
import navlie as nav
from pymlg import SE23, SO3
# import multinav
import multinav.arpimu  as arpimu
# import multinav.mrfilter
import numpy as np
from .utils import tril_to_vec, covariance_to_cholvec, cholvec_to_covariance
import pickle
import os 
from typing import Dict, List
from torch.utils.data import Dataset
import pandas as pd
from multinav.states import ARPIMUState

arpimu_identity_state = ARPIMUState.from_absolute_states(
    [
        nav.lib.IMUState(
            SE23.identity(), [0, 0, 0], [0, 0, 0], state_id=i
        )
        for i in range(3)
    ],
    1,
)

def _navlie_to_dataframes(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    

    Parameters
    ----------
    file_path : str

    Returns
    -------
    df_dict : dict[str, pd.DataFrame]
    """
    results = pickle.load(open(file_path, "rb"))


    # assumes results is a dict
    df_dict = {}
    for agent_name, agent_results in results.items():
        agent_results: nav.GaussianResultList = agent_results

        cov = agent_results.covariance
        states = np.array(
            [
                x.minus(arpimu_identity_state).ravel()
                for x in agent_results.state
            ]
        )
        cholvec = covariance_to_cholvec(torch.tensor(cov))
        states = torch.tensor(states)
        data_np = torch.concat((cholvec, states), dim=1).numpy()
        df_dict[agent_name] = pd.DataFrame(data_np)

    return df_dict

def _get_data_files(filename_list: str, data_dir: str) -> List[str]:
    """
    Identifies the list of files that need to be loaded from the cache to 
    constitute the dataset. If the cache files does not exist, then the
    navlie file is parsed and the cache files are created.

    Parameters
    ----------
    filename_list : str
        list of navlie files to load
    data_dir : str
        string of the directory where the navlie files are stored

    Returns
    -------
    List[str]
        list of cache files to load
    """
    #cache dir relative to this file
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")

    cache_file_list = os.listdir(cache_path)
    files = []
    for filename in filename_list:
        file_name_no_ext = os.path.splitext(filename)[0]

        if any([file_name_no_ext in f for f in cache_file_list]):
            # Then there is cache entry! just load it
            files.extend([f for f in cache_file_list if file_name_no_ext in f])
        else: 
            # Otherwise, need to parse the navlie file directly.
            df_dict = _navlie_to_dataframes(os.path.join(data_dir, filename))
            # Cache the results
            for agent_name, df in df_dict.items():
                cache_file_name = f"{file_name_no_ext}_{agent_name}.cache"
                df.columns = df.columns.astype(str)
                df.to_feather(os.path.join(cache_path, cache_file_name))
                files.append(cache_file_name)

    return files
        
        
    


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
    scaling_matrix =  _s @ _s.T 
    #scaling_matrix = torch.ones((45,45))

    def __init__(self, results_file_list, data_dir = None, samples_per_file=-1, scaling=True):


        self.files = _get_data_files(results_file_list, data_dir)

        cache_path = os.path.join(os.path.dirname(__file__), "../cache")
        
        dfs = []
        for file in self.files:
            df = pd.read_feather(os.path.join(cache_path, file))
            if samples_per_file >0 and samples_per_file < len(df):
                df = df.iloc[:samples_per_file, :]

            
            cholvec = torch.Tensor(df.iloc[:, :-45].values)
            cov = cholvec_to_covariance(cholvec, 45)
            cov = cov / self.scaling_matrix
            # cov = torch.linalg.inv(cov)
            # df.iloc[:, :-45] = tril_to_vec(cov).detach().numpy()
            df.iloc[:, :-45] = covariance_to_cholvec(cov).detach().numpy()

            dfs.append(df)

        # merge the dataframes 
        dfs = pd.concat(dfs, ignore_index=True)
        data = dfs.to_numpy()
        self.data = torch.Tensor(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return  self.data[idx, :-45],  self.data[idx, -45:]


    




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
                imu_list = imu_data.to_navlie()

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