"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `openstereo/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_val(model)
"""
from abc import abstractmethod, ABCMeta

import torch
from torch import nn

from utils import get_msg_mgr, is_dict, get_attr_from, is_list, get_valid_args
from . import backbone as backbones
from . import cost_processor as cost_processors
from . import disp_processor as disp_processors
from .loss_aggregator import LossAggregator


# from data.stereo_dataset import StereoDataset

class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, inputs) -> bool:
        """Do one forward step."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Compute the loss."""
        raise NotImplementedError

    # @abstractmethod
    # def train_step(self, loss_num) -> bool:
    #     """Do one training step."""
    #     raise NotImplementedError

    #
    # @abstractmethod
    # def inference(self, *args, **kwargs):
    #     """Do inference (calculate features.)."""
    #     raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    def __init__(self, cfg, device, is_train=True):
        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfg = cfg
        self.model_cfg = cfg['model_cfg']
        self.is_train = is_train
        self.msg_mgr.log_info(self.model_cfg)
        self.DispProcessor = None
        self.CostProcessor = None
        self.Backbone = None
        self.loss_fn = None
        self.build_network()
        self.build_loss_fn()
        self.device = device

    def build_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg) for cfg in backbone_cfg])
            return Backbone
        raise ValueError("Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_cost_processor(self, cost_processor_cfg):
        """Get the backbone of the model."""
        if is_dict(cost_processor_cfg):
            CostProcessor = get_attr_from([cost_processors], cost_processor_cfg['type'])
            valid_args = get_valid_args(CostProcessor, cost_processor_cfg, ['type'])
            return CostProcessor(**valid_args)
        if is_list(cost_processor_cfg):
            CostProcessor = nn.ModuleList([self.get_cost_processor(cfg) for cfg in cost_processor_cfg])
            return CostProcessor
        raise ValueError("Error type for -Cost-Processor-Cfg-, supported: (A list of) dict.")

    def build_disp_processor(self, disp_processor_cfg):
        """Get the backbone of the model."""
        if is_dict(disp_processor_cfg):
            DispProcessor = get_attr_from([disp_processors], disp_processor_cfg['type'])
            valid_args = get_valid_args(DispProcessor, disp_processor_cfg, ['type'])
            return DispProcessor(**valid_args)
        if is_list(disp_processor_cfg):
            DispProcessor = nn.ModuleList([self.get_cost_processor(cfg) for cfg in disp_processor_cfg])
            return DispProcessor
        raise ValueError("Error type for -Disp-Processor-Cfg-, supported: (A list of) dict.")

    def build_network(self):
        model_cfg = self.model_cfg
        if 'backbone_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['backbone_cfg'])
            self.Backbone = self.build_backbone(cfg)
        if 'cost_processor_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['cost_processor_cfg'])
            self.CostProcessor = self.build_cost_processor(cfg)
        if 'disp_processor_cfg' in model_cfg.keys():
            base_config = model_cfg['base_config']
            cfg = base_config.copy()
            cfg.update(model_cfg['disp_processor_cfg'])
            self.DispProcessor = self.build_disp_processor(cfg)
        if "init_parameters" in model_cfg.keys():
            self.init_parameters()

    def build_loss_fn(self, ):
        """Get the loss function."""
        loss_cfg = self.cfg['loss_cfg']
        self.loss_fn = LossAggregator(loss_cfg)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)
        return disp_out

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def prepare_inputs(self, inputs):
        """Reorganize input data for different models

        Args:
            inputs: the input data.
        Returns:
            dict: training data including ref_img, tgt_img, disp image,
                  and other meta data.
        """
        processed_inputs = {
            'ref_img': inputs['left'],
            'tgt_img': inputs['right'],
        }
        if 'disp' in inputs.keys():
            disp_gt = inputs['disp']
            # compute the mask of valid disp_gt
            max_disp = self.model_cfg['base_config']['max_disp']
            mask = (disp_gt < max_disp) & (disp_gt > 0)
            processed_inputs.update({
                'disp_gt': disp_gt,
                'mask': mask,
            })
        for k in ['pad_top', 'pad_bottom', 'pad_left', 'pad_right']:
            if k in inputs.keys():
                processed_inputs[k] = inputs[k]
        for k, v in processed_inputs.items():
            processed_inputs[k] = v.to(self.device) if torch.is_tensor(v) else v
        return processed_inputs

    def forward_step(self, batch_data) -> bool:
        batch_inputs = self.prepare_inputs(batch_data)
        outputs = self.forward(batch_inputs)
        return outputs

    def compute_loss(self, inputs, outputs):
        """Compute the loss."""
        training_disp = outputs['training_disp']
        loss, _ = self.loss_fn(training_disp)
        return loss

    def train_step(self, data_batch, optimizer):
        """Train the model for one step."""
        self.train()
        optimizer.zero_grad()
        outputs = self.forward_step(data_batch)
        loss = self.compute_loss(None, outputs)
        loss.backward()
        optimizer.step()
        return loss

    def val_step(self, data_batch):
        """Validate the model for one step."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward_step(data_batch)
            loss = self.compute_loss(None, outputs)
        return loss

    def fix_bn(self):
        """Fix the batch normalization layers."""
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()
