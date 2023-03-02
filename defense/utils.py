import torch, os, json
import numpy as np
from torch import Tensor

def tanh_func(x: Tensor) -> Tensor:
    return x.tanh().add(1).mul(0.5)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':    # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures


def to_numpy(x, **kwargs) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)


def read_json(path):
    with open(path) as f:
        return json.load(f)
    

def write_json(path, data: dict, update=False):
    data_w = data
    if update is True:
        info: dict = read_json(path)
        data_w = info.update(data_w)
    with open(makedir(path), 'w') as f:
        f.write(json.dumps(data_w, indent=1))


def makedir(path: str) -> str:
    dirname = os.path.dirname(path)
    if os.path.exists(dirname) is False:
        os.makedirs(dirname)
    return path
