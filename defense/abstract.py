import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from logging import Logger
from utils import tools
from torch import nn, Tensor
from typing import Callable, Tuple

class BackdoorDefense(ABC):
    def __init__(
            self, 
            model: nn.Module, 
            transform: Callable[[Tensor], Tensor],
            input_shape: Tuple[int, int, int],
            dataset: Dataset, 
            log_path: str, 
            logger: Logger,
            device: str = 'cuda') -> None:
        """The backdoor defense method

        Args:
            model (nn.Module): The suspicious PyTorch model.
            transform (Callable[[Tensor], Tensor]): The callable transform (including normalization).
            input_shape (Tuple[int, int, int]): Then model input shape [in_channel, H, W].
            dataset (Dataset): The unnormalized dataset.
            log_path (str): The folder used to store the log files.
            logger (Logger): The logger.
            device (str, optional): CPU/GPU. Defaults to 'cuda'.
        """        
        super().__init__()
        self.model: nn.Module = model
        self.transform = transform
        self.input_shape = input_shape
        self.dataset = dataset
        self.logger = logger
        self.log_path = os.path.join(log_path, tools.timestr())
        logger.info(f'All log files will be saved in {self.log_path}')
        self.device = device
        if os.path.exists(self.log_path) is False:
            os.makedirs(self.log_path)

    @abstractmethod
    def detect():
        pass


