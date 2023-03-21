import os, datetime
from abc import ABC, abstractmethod
from logging import Logger
from dataclasses import dataclass


@dataclass
class DefenseConfig(ABC):
    pass


class BackdoorDefense(ABC):
    def __init__(
            self,  
            log_path: str, 
            logger: Logger,
            cfg: DefenseConfig = None,
            device: str = 'cuda') -> None:
        """The backdoor defense method

        Args:
            log_path (str): The folder used to store the log files.
            logger (Logger): The logger.
            device (str, optional): CPU/GPU. Defaults to 'cuda'.
        """        
        super().__init__()
        self.logger = logger
        self.log_path = os.path.join(log_path, f'{datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")}')
        self.cfg = cfg
        logger.info(f'All log files will be saved in {self.log_path}')
        logger.info(f'Config received: {str(cfg)}')
        self.device = device
        if os.path.exists(self.log_path) is False:
            os.makedirs(self.log_path)

    @abstractmethod
    def detect():
        pass


