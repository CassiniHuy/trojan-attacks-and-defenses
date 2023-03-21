import torch
import os
import time
import datetime
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch import Tensor, nn
from dataclasses import dataclass
from logging import Logger
from typing import Tuple, Callable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from .abstract import BackdoorDefense, DefenseConfig
from .utils import AverageMeter, tanh_func, normalize_mad, to_numpy, write_json

@dataclass
class NeuralCleanseConfig(DefenseConfig):
    epoch: int = 2
    lr: float = 0.002
    betas: Tuple[float] = (0.5, 0.9)
    weight_decay: float = 0.0005
    lr_decay_ratio: float = 0.2
    batch_size: int = 64

    init_cost: float = 1e-3
    cost_multiplier: float = 1.5
    cost_multiplier_up: float = 1.5
    cost_multiplier_down: float = 1.5 ** 1.5
    patience: float = 10
    attack_succ_threshold: float = 0.99
    early_stop: bool = True
    early_stop_threshold: float = 0.99
    early_stop_patience: float = 10 * 2


class NeuralCleanse(BackdoorDefense):
    name: str = 'neural_cleanse'
    
    def __init__(
            self, 
            model: nn.Module, 
            transform: Callable[[Tensor], Tensor], 
            input_shape: Tuple[int, int, int],
            dataset: Dataset, 
            log_path: str, logger: Logger, 
            cfg: NeuralCleanseConfig = None,
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
        self.model = model
        self.transform = transform
        self.input_shape = input_shape
        self.dataset = dataset
        cfg = cfg if cfg else NeuralCleanseConfig()
        super().__init__(log_path, logger, cfg, device)

    def detect(self):
        self.result = {}
        self.result["FinalResult"] = {}
        self.result["intermediate"] = {}
        mark_list, mask_list, loss_list = self.get_potential_triggers()
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1).tolist()
        mad = normalize_mad(mask_norms).tolist()
        loss_mad = normalize_mad(loss_list).tolist()
        loss_list = loss_list.tolist()
        self.logger.info('mask norms: ', mask_norms)
        self.logger.info('mask MAD: ', mad)
        self.logger.info('loss: ', loss_list)
        self.logger.info('loss MAD: ', normalize_mad(loss_list))
        self.result["FinalResult"]["mask_norms"] = " ｜ ".join(map(str, mask_norms))
        self.result["FinalResult"]["mask_MAD"] = ' ｜ '.join(map(str, mad))
        self.result["FinalResult"]["loss"] = ' ｜ '.join(map(str, loss_list))
        self.result["FinalResult"]["loss_MAD"] = ' ｜ '.join(map(str, loss_mad))

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        mark_list = [to_numpy(i) for i in mark_list]
        mask_list = [to_numpy(i) for i in mask_list]
        loss_list = [to_numpy(i) for i in loss_list]
        
        suspicious_labels = []
        for i in range(len(mad)):
            if mad[i] > 2:
                suspicious_labels.append(i)
        
        self.logger.info('suspicious_labels: ',suspicious_labels)
        self.result["FinalResult"]["suspicious_lables"] = ' | '.join(map(str, suspicious_labels))

        if len(suspicious_labels) == 0:
            self.logger.info('cannot find suspicious label')

        write_json(os.path.join(self.log_path, 'result.json'), self.result)
    
    
    def convert_to_visual_image(self, mark: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        background = torch.zeros_like(mark, device=mark.device)
        trigger = background + mask * (mark - background) # = (1 - mask) + mask * mark
        return trigger
        

    def get_potential_triggers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        module = self.model
        mark_list, mask_list, loss_list = [], [], []
        mark_list_save, mask_list_save = [], []
        criterion = nn.CrossEntropyLoss()
        remask_path = os.path.join(self.log_path, 'remask.npz')
        self.logger.info(f'Reversed marks of numpy format will be saved in {remask_path}')
        for label in range(module.num_classes):
            length = len(str(module.num_classes))
            self.logger.info(f'Reversing class {label}/{module.num_classes}')
            str1 = "Class: [{0:s} / {1:d}]".format(str(label).rjust(length), module.num_classes)
            inter_result = []
            self.result["intermediate"][str1] = inter_result
            mark, mask, loss = self.remask(label, str1, inter_result, criterion)
            self.result["intermediate"][str1] = "<br/>".join(map(str, inter_result))
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)

            mark_list_save.append(np.array(mark.cpu()))
            mask_list_save.append(np.array(mask.cpu()))

            np.savez(remask_path, mark_list=mark_list_save, mask_list=mask_list_save, loss_list=loss_list)
            trigger = self.convert_to_visual_image(mark, mask)
            trigger_savepath = os.path.join(self.log_path, f'trigger_{label}.png')
            save_image(trigger, trigger_savepath)
            self.logger.info(f'Trigger image saved at {trigger_savepath}')
            self.logger.info('Defense results saved at: ' + remask_path)
        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)
        return mark_list, mask_list, loss_list

    def remask(self, label: int, str1: str, inter_result: list, criterion):
        self.model.to(self.device)
        self.model.eval()
        atanh_mark = torch.randn(self.input_shape, device=self.device)
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.input_shape[-2:], device=self.device)
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=self.cfg.lr, betas=self.cfg.betas)
        optimizer.zero_grad()

        cost = self.cfg.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        trainloader = DataLoader(self.dataset,
                                 batch_size=self.cfg.batch_size,
                                 shuffle=True,
                                 pin_memory=True)

        for _epoch in range(self.cfg.epoch):
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            trainloader = tqdm(trainloader)

            for (_input, _label) in trainloader:
                batch_size = _label.size(0)
                _input, _label = _input.to(self.device), _label.to(self.device)
                X = _input + mask * (mark - _input) # = (1 - mask) + mask * mark
                Y = label * torch.ones_like(_label, dtype=torch.long)
                
                _output = self.model(self.transform(X))

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = criterion(_output, Y)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)    # (h, w)
                mark = tanh_func(atanh_mark)    # (c, h, w)
                trainloader.set_description_str(f'BinaryEntropy: {batch_loss:.4f}')
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = "Epoch: [{0:s} / {1:d}] ---> ".format(str(_epoch+1), self.cfg.epoch)
            _str = ' '.join([
                f"Loss: {losses.avg:.4f},",
                f"Acc: {acc.avg:.2f}, ",
                f"Norm: {norm.avg:.4f},",
                f"Entropy: {entropy.avg:.4f},",
                f"Time: {epoch_time},",
            ])
            str2 = pre_str + _str
            inter_result.append(str2)
            if acc.avg >= self.cfg.attack_succ_threshold and norm.avg < norm_best:
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.cfg.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.cfg.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.cfg.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.cfg.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.cfg.patience:
                    cost = self.cfg.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.cfg.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.cfg.patience:
                cost_up_counter = 0
                cost *= self.cfg.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.cfg.patience:
                cost_down_counter = 0
                cost /= self.cfg.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = tanh_func(atanh_mask).detach()
                mark_best = tanh_func(atanh_mark).detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        self.result["intermediate"][str1] = inter_result
        return mark_best, mask_best, entropy_best

