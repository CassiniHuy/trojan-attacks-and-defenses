import torch, os, json, cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from typing import Tuple, List, Union

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


def load_image(path):
    return Image.open(path).convert('RGB')


def load_image_tensor(path: Union[str, List[str]],
                      size: Tuple[int] = None,
                      norm: Tuple[Tuple[float], Tuple[float]] = None
                      ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Load image tensor from paths.

    Args:
        path (Union[str, List[str]]): Path str or list of paths.
        size (Tuple[int], optional): Model input size. Defaults to None.
        norm (Tuple[Tuple[float], Tuple[float]], optional): mean and std. Defaults to None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Tensor or list of tensor, shape=[1, C, H, W].
    """    
    paths = path if isinstance(path, list) else [path]
    tensors = list()
    for img_path in paths:
        img = load_image(img_path)
        if size is not None:
            img = transforms.Resize(size)(img)
        img_tensor: Tensor = transforms.ToTensor()(img)
        if norm is not None:
            img_tensor = transforms.Normalize(norm[0], norm[1])(img_tensor)
        tensors.append(img_tensor.unsqueeze(dim=0))
    tensors = tensors if isinstance(path, list) else tensors[0]
    return tensors


def gen_checker_pattern(size: Tuple[int] = (224, 224), bins: int = 15, channels: int = 3) -> Tensor:
    """Generate checker pattern.

    Args:
        size (Tuple[int], optional): Image size. Defaults to (224, 224).
        bins (int, optional): Pattern bins. Defaults to 15.
        channels (int, optional): Image channels. Defaults to 3.

    Returns:
        Tensor: Image tensor of shape=[1, C, H, W].
    """    
    pattern = torch.zeros([1, channels, *size])
    bin_length = int(size[0] / bins)
    assert bin_length > 0
    for i in range(size[0]):
        for j in range(size[1]):
            if i % (bin_length * 2) < bin_length: 
                if j % (bin_length * 2) < bin_length:
                    pattern[:,:,i,j] = 1
                else:
                    pattern[:,:,i,j] = 0
            else:
                if j % (bin_length * 2) < bin_length:
                    pattern[:,:,i,j] = 0
                else:
                    pattern[:,:,i,j] = 1
    return pattern


def gen_heatmap(img: Tensor, mask: Tensor, save_path: str):
    im = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    hm = mask[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    hm = cv2.applyColorMap(np.uint8(hm), cv2.COLORMAP_JET)
    cam = cv2.addWeighted(np.uint8(im), 0.5, hm, 0.5, 0)
    if os.path.exists(os.path.dirname(save_path)) is False:
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, cam)
