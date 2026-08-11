from abc import ABC, abstractmethod

import torch

from src.configs import DataConfigs


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        self.data_configs = data_configs
        self.data_dir = data_configs.data_dir

        self.num_samples = data_configs.num_samples

        self.kwargs = kwargs

    @abstractmethod
    def parse_data(self):
        pass
