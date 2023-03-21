import warnings, torch, os, json
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, nn
from collections import OrderedDict
from dataclasses import dataclass
from logging import Logger
from typing import Callable, List, Tuple, Union
from captum.attr import LayerGradCam
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from selectivesearch import selective_search
from tqdm import tqdm
from .abstract import DefenseConfig, BackdoorDefense
from .utils import load_image_tensor, gen_checker_pattern

GAUSSIAN_NOISE = 1
CHECKER_PATTERN = 2

'''
To know the meaning of the config and code, please refer to the paper: 
https://arxiv.org/abs/1812.00292
'''

@dataclass
class SentiNetConfig(DefenseConfig):
    """Default SentiNet Config.
    """    
    benign_img_num: int = 400
    test_img_num: int = 100
    adv_img_num: int = 400

    enable_class_proposal: bool = True
    batch_size: int = 100
    selective_search_scale: int = 500
    selective_search_sigma: float = .9
    selective_search_minsize: int = 10
    selective_search_max_regions: int = 100
    class_proposal_k: int = 2

    mask_threshold: float = .15
    inert_type: int = GAUSSIAN_NOISE
    inert_noise_mean: float = .0
    inert_noise_std: float = 1.0
    inert_checker_bins: int = 15

    x_interval_size: float = .02


class QuadraticModel():
    def __init__(self) -> None:
        self.poly_feat = PolynomialFeatures(degree=2, include_bias=False)
        self.linear_model = LinearRegression()

    def _poly_feat(self, x: List[List[float]]) -> List[List[float]]:
        return self.poly_feat.fit_transform(x)

    def fit(self, x: List[List[float]], y: List[float]):
        outpts_feats = self._poly_feat([[item] for item in x])
        self.linear_model.fit(outpts_feats, y)

    def __call__(self, x: float) -> float:
        return self.linear_model.predict(self.poly_feat.fit_transform([[x]]))[0]


class SentiNet(BackdoorDefense):
    def __init__(self, 
                 model: nn.Module,
                 layer: nn.Module,
                 input_size: Tuple[int, int],
                 transform: Callable[[Tensor], Tensor],
                 benign_imgs: List[str],
                 test_imgs: List[str],
                 adv_imgs: List[str],
                 log_path: str, 
                 logger: Logger, 
                 cfg: SentiNetConfig = None, 
                 device: str = 'cuda') -> None:
        """_summary_

        Args:
            model (nn.Module): The model.
            layer (nn.Module): The model layer used to compute gradcam.
            input_size (Tuple[int]): Input size of the model.
            transform (Callable[[Tensor], Tensor]): Transform for the model input. logits = model(transform(x))
            benign_imgs (List[str]): Benign image paths.
            test_imgs (List[str]): Benign test image paths.
            adv_imgs (List[str]): Adversarial image paths.
            log_path (str): The log path used to save log files.
            logger (Logger): The logger.
            cfg (SentiNetConfig, optional): Config. Defaults to None.
            device (str, optional): cuda/cpu. Defaults to 'cuda'.
        """        
        self.model = model
        self.layer = layer
        self.input_size = input_size
        self.transform = transform
        self.benign_imgs = benign_imgs
        self.test_imgs = test_imgs
        self.adv_imgs = adv_imgs
        self.cfg: SentiNetConfig
        cfg = cfg if cfg else SentiNetConfig()
        super().__init__(log_path, logger, cfg, device)
        self._init_model()
        self._init_inert()
    
    def _init_model(self):
        self.model.eval()
        self.model.to(self.device)
        self.gradcam = LayerGradCam(self._model_forward, layer=self.layer)
        self.logger.debug(f'INIT: GradCAM loaded on the layer {str(self.layer)}.')
    
    def _init_inert(self):
        if self.cfg.inert_type == CHECKER_PATTERN:
            self.inert_image = gen_checker_pattern(
                size=self.input_size, bins=self.cfg.inert_checker_bins).to(self.device)
            self.logger.debug(f'INIT: Insert type set as checker pattern with '
                              f'size {self.input_size} and {len(self.cfg.inert_checker_bins)} bins')
        elif self.cfg.inert_type == GAUSSIAN_NOISE:
            self.inert_image = torch.normal(
                mean=torch.ones((1, 3, *self.input_size)) * self.cfg.inert_noise_mean, 
                std=self.cfg.inert_noise_std).to(self.device)
            self.logger.debug(f'INIT: Insert type set as gaussian noise with '
                              f'mean {self.cfg.inert_noise_mean} and std {self.cfg.inert_noise_std}')
        else:
            raise ValueError(f'Invalid configuration: inert_type={self.cfg.inert_type}')
                
    def _model_forward(self, x: Tensor) -> Tensor:
        return self.model(self.transform(x))
    
    def _model_predict(self, x: Union[Tensor, List[Tensor]]
                       ) -> Union[Tuple[float, int], List[Tuple[float, int]]]:
        """Model predict (batch support)

        Args:
            x (Union[Tensor, List[Tensor]]): 
                tensor or list of tensor, shape=[1, C, H, W].

        Returns:
            Union[Tuple[float, int], List[Tuple[float, int]]]: 
                Tuple of list of tuple, of confidence and label.
        """        
        if isinstance(x, Tensor):
            output = torch.softmax(self._model_forward(x), dim=-1)
            top1 = torch.topk(output, k=1)
            return top1[0].item(), top1[1].item()
        else:
            outputs = list()
            for i in range(int(np.ceil(len(x) / self.cfg.batch_size))):
                batch = torch.cat(x[i * self.cfg.batch_size: (i + 1) * self.cfg.batch_size], dim=0)
                output = torch.softmax(self._model_forward(batch.to(self.device)), dim=-1).detach().cpu()
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            values, indices = torch.topk(outputs, k=1)
            values, indices = values.view(-1).numpy().tolist(), indices.view(-1).numpy().tolist()
            return list(zip(values, indices))
    
    ''' Algorithm 1: Class proposal. 
    '''
    def _selective_search(self, x: Tensor) -> List[Tensor]:
        """Selective search on the input tensor image.

        Args:
            x (Tensor): tensor of shape [1, C, H, W].

        Returns:
            List[Tensor]: List of proposal, tensor shape=[1, C, H, W].
        """        
        x_np: np.ndarray = x[0].permute(1, 2, 0).to('cpu').numpy() * 255
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, regions = selective_search(x_np.astype(np.uint8), 
                                        scale=self.cfg.selective_search_scale,
                                        sigma=self.cfg.selective_search_sigma,
                                        min_size=self.cfg.selective_search_minsize)
        regions = regions[:self.cfg.selective_search_max_regions]
        proposals = list()
        for region in regions:
            x, y, w, h = region['rect']
            if w * h == 0:
                continue
            prop_region = x_np[x:x+w, y:y+h, :]
            prop_region = torch.tensor(prop_region).to(self.device) / 255
            prop_region = prop_region.permute(2, 0, 1).unsqueeze(dim=0)
            prop_region = nn.functional.interpolate(prop_region, size=self.input_size)
            proposals.append(prop_region)
        return proposals
    
    def _top_conf(self, c: List[Tuple[float, int]], primary_c: int) -> List[int]:
        """Top confidence of the prediction.

        Args:
            c (List[Tuple[float, int]]): List of tuple containing confidence and label.
            primary_c (int): Primary class of the input tensor.

        Returns:
            List[int]: List of proposal classes.
        """        
        c_filtered = filter(lambda xy: xy[1] != primary_c, c)
        c_sorted = sorted(c_filtered, key=lambda xy: xy[0], reverse=True)
        c_ordered_dict = OrderedDict()
        for _, c in c_sorted:
            c_ordered_dict[c] = None
        c_topk = list(c_ordered_dict.keys())[:self.cfg.class_proposal_k]
        return c_topk
    
    def class_proposal(self, imgs: List[Tensor]) -> List[Tuple[int, List[int]]]:
        """Algorithm 1: Class proposal.

        Args:
            imgs (List[Tensor]): List of image tensors, shape=[1, C, H, W].

        Returns:
            List[Tuple[int, List[int]]]: 
                List of proposals, containing primary class and proposal classes.
        """        
        self.logger.debug(f'Algorithm 1: class proposal...')
        classes_prop = list()
        imgs = tqdm(imgs)
        for i, img in enumerate(imgs):
            props = self._selective_search(img)
            if len(props) == 0:
                self.logger.warning(f'Algorithm 1: Zero class proposal for {i}th image.')
                c = list()
            else:
                c = self._model_predict(props)
            primary_c = self._model_predict(img)[1]
            classes_prop.append((primary_c, self._top_conf(c, primary_c)))
        return classes_prop
    
    ''' Algorithm 2: Mask Generation. 
    '''
    def _mask_gradcam(self, x: Tensor, c: int) -> Tensor:
        """Generate mask using gradcam.

        Args:
            x (Tensor): Image tensor of shape [1, C, H, W].
            c (int): Class label.

        Returns:
            Tensor: Mask of shape [1, C, H, W].
        """        
        x = x.clone()
        x.requires_grad_(True)
        attr = self.gradcam.attribute(x, c)
        top_k = max(1, int(len(attr.view(-1)) * self.cfg.mask_threshold)) # if attr.shape is too small
        threshold = torch.topk(
            attr.view(-1), 
            k=top_k)[0][-1]
        mask = (attr > threshold).float()
        mask = torch.nn.functional.interpolate(mask, size=x.shape[-2:], mode='nearest')
        return mask
    
    def mask_generate(self, benign_imgs: List[Tensor], 
                        classes_prop: List[Tuple[int, List[int]]]
                        ) -> List[Tuple[int, List[Tensor]]]:
        """Algorithm 2: Mask generation.

        Args:
            benign_imgs (List[Tensor]): Benign image tensors of shape [1, C, H, W].
            classes_prop (List[Tuple[int, List[int]]]): 
                Class proposals containing primary class and proposed classes.

        Returns:
            List[Tuple[int, List[Tensor]]]: List of masks, containing primary class and masks.
        """        
        self.logger.debug(f'Algorithm 2: Mask generation...')
        assert len(benign_imgs) == len(classes_prop)
        masks = list()
        progress = tqdm(range(len(benign_imgs)))
        for i in progress:
            img = benign_imgs[i]
            primary_c, classes = classes_prop[i]
            mask_y = self._mask_gradcam(img, primary_c)
            masks_c = [self._mask_gradcam(img, c) for c in classes]
            for j in range(len(masks_c)):
                mask = mask_y.clone()
                mask[masks_c[j] == 1] = 0
                masks_c[j] = mask
            masks.append((primary_c, masks_c + [mask_y]))
        return masks
    
    ''' Algorithm 3: Testing. 
    '''
    def _inert_pattern(self, masks: List[Tensor]) -> List[Tensor]:
        """Generate inert patterns.

        Args:
            masks (List[Tensor]): List of masks of shape [1, C, H, W].

        Returns:
            List[Tensor]: List of pettern tensors of shape [1, C, H, W].
        """        
        patterns = [self.inert_image * mask for mask in masks]
        return patterns
    
    def _overlay(self, inputs: List[Tensor], regions: List[Tensor]) -> List[Tensor]:
        """Overlay the regions into the inputs.

        Args:
            inputs (List[Tensor]): Image tensors of shape [1, C, H, W].
            regions (List[Tensor]): Region tensors of shape [1, C, H, W].

        Returns:
            List[Tensor]: Overlaided images of shape [1, C, H, W].
        """        
        overlaid = list()
        for ipt in inputs:
            for region in regions:
                ipt = ipt.clone()
                ipt[region != 0] = 0
                ipt += region
                overlaid.append(ipt)
        return overlaid
    
    def testing(self, imgs: List[Tensor], test_imgs: List[Tensor], 
                masks: List[Tuple[int, List[Tensor]]]) -> Tuple[List[float], List[float]]:
        """Algorithm 3: Testing.

        Args:
            imgs (List[Tensor]): Images of shape [1, C, H, W].
            test_imgs (List[Tensor]): Test images of shape [1, C, H, W].
            masks (List[Tuple[int, List[Tensor]]]): List containing primary class and masks.

        Returns:
            Tuple[List[float], List[int]]: Avgrage confidence ip, fooled percentage
        """        
        self.logger.debug('Algorithm 3: Testing...')
        assert len(imgs) == len(masks)
        fooleds, avg_conf_ips = list(), list()
        progress = tqdm(range(len(imgs)))
        for i in progress:
            primary_c, masks_c = masks[i]
            # Calculate avgconfip
            ip = self._inert_pattern(masks_c)
            x_ip = self._overlay(test_imgs, ip)
            x_ip_c = self._model_predict(x_ip)
            avg_conf_ip = sum([xy[0] for xy in x_ip_c]) / len(x_ip_c)
            avg_conf_ips.append(avg_conf_ip)
            # Calculate fooled
            r = [imgs[i] * mask for mask in masks_c]
            x_r = self._overlay(test_imgs, r)
            x_r_c = self._model_predict(x_r)
            fooled = sum([xy[1] == primary_c for xy in x_r_c]) / len(x_r_c)
            fooleds.append(fooled)
        return avg_conf_ips, fooleds
    
    ''' Algorithm 4: Decision Boundary. 
    '''
    def _out_points(self, avg_conf_ip: List[float], fooled_yr: List[float]) -> List[Tuple[float, int]]:
        """Collect outer points from the given points.

        Args:
            avg_conf_ip (List[float]): Avarage confidence of IP.
            fooled_yr (List[float]): Fooled percentages.

        Raises:
            RuntimeError: No points found.

        Returns:
            List[Tuple[float, int]]: Points containing avg_conf and fooled.
        """        
        assert len(fooled_yr) == len(avg_conf_ip)
        # Divide points into n bins
        conf_max, conf_min = max(avg_conf_ip), min(avg_conf_ip)
        n_bins = int(np.ceil((conf_max - conf_min) / self.cfg.x_interval_size))
        self.logger.debug(f'Algorithm 4: Points min and max confidence: {conf_min:.4f}, {conf_max:.4f} and {n_bins} bins.')
        assert n_bins > 0
        points_in_bins = {bin_id: list() for bin_id in range(n_bins)}
        # Put points to corresponding bins
        for i in range(len(fooled_yr)):
            bin_id = int((avg_conf_ip[i] - conf_min) / self.cfg.x_interval_size)
            points_in_bins[bin_id].append((avg_conf_ip[i], fooled_yr[i]))
        points_in_bins = filter(lambda x: len(x) != 0, points_in_bins.values())
        outpts = [max(points, key=lambda xy: xy[1]) for points in points_in_bins]
        self.logger.debug(f'Algorithm 4: {len(outpts)} out points found.')
        if len(outpts) == 0:
            raise RuntimeError(f'No out points can be found.')
        return outpts
    
    def _approximate_curve(self, outpts: List[Tuple[float, float]]) -> Callable[[float], float]:
        """Appromixate a quadratic curve.

        Args:
            outpts (List[Tuple[float, float]]): Outer points.

        Returns:
            Callable[[float], float]: The quadratic curve.
        """        
        quad_model = QuadraticModel()
        quad_model.fit([xy[0] for xy in outpts], [xy[1] for xy in outpts])
        return quad_model
    
    def _cobyla(self, x: float, y: float, f_curve: Callable[[float], float]) -> float:
        """Calculate the distrance to the curve using COBYLA method.

        Args:
            x (float): avg_conf
            y (float): fooled
            f_curve (Callable[[float], float]): The curve function.

        Returns:
            float: The distance.
        """        
        distance_sq = lambda ipt: (ipt - x) ** 2 + (f_curve(x) - y) ** 2
        result = minimize(distance_sq, (x,), method='cobyla')
        return np.sqrt(result.fun)
    
    def decision_boundary(self, avg_conf_ip: List[float], fooled_yr: List[float]
                          ) -> Tuple[Callable[[float], float], float]:
        """Algorithm 4: Decision Boundary.

        Args:
            avg_conf_ip (List[float]): List of avg_conf.
            fooled_yr (List[float]): List of fooled.

        Returns:
            Tuple[Callable[[float], float], float]: Curve function and the distance threshold.
        """        
        self.logger.debug(f'Algorithm 4: Decision boundary...')
        assert len(fooled_yr) == len(avg_conf_ip)
        outpts = self._out_points(avg_conf_ip, fooled_yr)
        f_curve = self._approximate_curve(outpts)
        avg_d = 0
        for x, y in zip(avg_conf_ip, fooled_yr):
            if f_curve(x) > y:
                avg_d += self._cobyla(x, y, f_curve)
        d = avg_d / len(fooled_yr) # In the paper, it devides the size of B
        self.logger.info(f'Algorithm 4: Quadratic curve trained with distance threshold {d:.6f}')
        return f_curve, d

    ''' Detect using four algorithms.
    '''
    def _log_results(self, 
                     x_train: List[float], y_train: List[int], 
                     x_test: List[float], y_test: List[int], 
                     f_curve: Callable[[float], float], d: float, 
                     y_pred: List[float], y_thr: List[float],):
        pred = [(y_pred[i] < y_test[i]) and (y_thr[i] > d) for i in range(len(y_thr))]
        self.logger.info(f'{sum(pred)}/{len(pred)} ({sum(pred)/len(pred):.4f}) '
                         'samples are predicted as adversarial samples.')
        self._log_figure(x_train, y_train, x_test, y_test, f_curve)
        self._log_file(x_train, y_train, x_test, y_test, f_curve, d, y_pred, y_thr)

    def _log_config(self):
        save_path = os.path.join(self.log_path, 'sentinet-config.json')
        with open(save_path, 'w') as f:
            f.write(json.dumps(self.cfg.__dict__, indent=1))
        self.logger.info(f'Config file saved into {save_path}')
    
    def _log_file(self, 
                  x_train: List[float], y_train: List[int], 
                  x_test: List[float], y_test: List[int], 
                  f_curve: QuadraticModel, d: float, 
                  y_pred: List[float], y_thr: List[float]):
        info = dict(
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            curve_coef=f_curve.linear_model.coef_.tolist(),
            curve_intercept=f_curve.linear_model.intercept_,
            d=d, y_pred=y_pred, y_thr=y_thr,
        )
        save_path = os.path.join(self.log_path, 'sentinet-results.json')
        with open(save_path, 'w') as f:
            f.write(json.dumps(info, indent=1))
        self.logger.info(f'Log file saved into {save_path}')

    def _log_figure(self, 
                    x_train: List[float], y_train: List[float], 
                    x_test: List[float], y_test: List[float],
                    curve: Callable[[float], float]):
        # Draw points
        plt.scatter(x_train, y_train, marker='o', color='blue', s=5, alpha=1.0, label='benign')
        plt.scatter(x_test, y_test, marker='o', color='red', s=5, alpha=1.0, label='adversarial')
        # Draw the cruve
        x_min, x_max = min(x_train), max(x_train)
        x_points = np.linspace(x_min, x_max).tolist()
        y_points = [curve(x) for x in x_points]
        plt.scatter(x_points, y_points, color='black', s=5, alpha=1.0, label='boundary')
        plt.xlabel("AvgConf")
        plt.ylabel("Fooled")
        plt.xlim([min(x_train + x_test), max(x_train + x_test)])
        plt.ylim([0, 1])
        plt.legend(loc='upper left')
        # Save figure
        save_path = os.path.join(self.log_path, 'sentinet-results.png')
        plt.savefig(save_path)
        self.logger.info(f'The figure is saved into {save_path}')

    def obtain_features(self, imgs: List[Tensor], test_imgs: List[Tensor]) -> Tuple[List[int], List[float]]:
        self.logger.info(f'Class proposal for {len(imgs)} images...')
        cls_proposals = self.class_proposal(imgs)
        self.logger.info(f'Mask generation for {len(imgs)} images...')
        masks = self.mask_generate(imgs, cls_proposals)
        self.logger.info(f'Testing for {len(imgs)} images on {len(test_imgs)} test images...')
        foolded, avg_conf_ip = self.testing(imgs, test_imgs, masks)
        assert len(foolded) == len(avg_conf_ip)
        return foolded, avg_conf_ip
    
    def train_on_benign(self, benign_imgs: List[Tensor], test_imgs: List[Tensor]
                        ) -> Tuple[List[float], List[float], Callable[[float], float], float]:
        self.logger.info(f'{len(benign_imgs)} benign images loaded.')
        x_train, y_train = self.obtain_features(benign_imgs, test_imgs)
        f_curve, d = self.decision_boundary(x_train, y_train)
        return x_train, y_train, f_curve, d
    
    def test_on_adv(self, f_curve: Callable[[float], float], 
                    adv_imgs: List[Tensor], test_imgs: List[Tensor]
                    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        self.logger.info(f'{len(adv_imgs)} adv images loaded.')
        x_test, y_test = self.obtain_features(adv_imgs, test_imgs)
        y_pred = [f_curve(x) for x in x_test]
        y_thr = [self._cobyla(x, y, f_curve) for x, y in zip(x_test, y_test)]
        return x_test, y_test, y_pred, y_thr
    
    def predict_on_feats(self, f_curve: Callable[[float], float], d: float, x: float, y: float) -> bool:
        pred = f_curve(x)
        y_d = self._cobyla(x, y, f_curve)
        return (pred < y) and (y_d > d)

    def detect(self) -> dict:
        self._log_config()
        test_imgs = load_image_tensor(
            self.test_imgs[:self.cfg.test_img_num], size=self.input_size, device=self.device)
        assert len(test_imgs) != 0
        self.logger.info(f'{len(test_imgs)} test images loaded.')
        # Train model
        self.logger.debug(f'Model training...')
        benign_imgs = load_image_tensor(
            self.benign_imgs[:self.cfg.benign_img_num], self.input_size, device=self.device)
        assert len(benign_imgs) != 0
        x_train, y_train, f_curve, d = self.train_on_benign(benign_imgs, test_imgs) 
        # Test model
        self.logger.debug(f'Model testing...')
        adv_imgs = load_image_tensor(
            self.adv_imgs[:self.cfg.adv_img_num], size=self.input_size, device=self.device)
        assert len(adv_imgs) != 0
        x_test, y_test, y_pred, y_thr = self.test_on_adv(f_curve, adv_imgs, test_imgs)
        # Log info
        self.logger.info(f'Tested on {len(y_thr)} suspicious samples.')
        self._log_results(x_train, y_train, x_test, y_test, f_curve, d, y_pred, y_thr)
        return dict(x_train=x_train, y_train=y_train, 
                    x_test=x_test, y_test=y_test, 
                    f_curve=f_curve, d=d, y_pred=y_pred, y_thr=y_thr,)
