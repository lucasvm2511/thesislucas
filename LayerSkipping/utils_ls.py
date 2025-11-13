import copy
import json
import random
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
home_dir = os.path.expanduser('~')  # This will return the home directory path (e.g., /home/username)
# Concatenate the home directory with the 'LayerSkipping/models' path
repo_dir = os.path.join(home_dir, 'workspace/CNAS')
import numpy as np
from ofa.utils.pytorch_utils import count_parameters

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchprofile import profile_macs
from torch.nn import Conv2d,ReLU,Linear,Sequential,Flatten,BatchNorm2d,AvgPool2d,MaxPool2d
from models.mobilenet_v3 import SkippingMobileNetV3

def get_device(model: nn.Module):
    return next(model.parameters()).device

def get_skipping_mobilenetv3(subnet, subnet_path, res, n_classes, gate_type="stable", temperature=1.0, enable_gates=True, gate_hidden_size=None, target_sparsities=None):
    """
    Create Skipping MobileNetV3 from a given subnet with per-block target sparsities controlling gate placement.
    
    Args:
        subnet: Base subnet (MobileNetV3 model)
        subnet_path: Path to subnet configuration file
        res: Input resolution
        n_classes: Number of classes
        gate_type: Type of gate to use ("stable" or "conv")
        temperature: Temperature for gates
        enable_gates: Whether to enable gates
        gate_hidden_size: Hidden layer size for gates (optional, from config if None)
        target_sparsities: Array of target sparsities per block (optional, from config if None)
                          0 = no gate, 0.3/0.5/0.7 = sparsity target
    
    Returns:
        backbone: SkippingMobileNetV3 model
    """
    import json
    
    # Load subnet configuration
    config = json.load(open(subnet_path))
    
    # Extract configuration parameters
    depth = config['d']  # depth array
    
    # Extract gate parameters if available
    config_gate_hidden_size = config.get('gate_hidden_size', None)
    config_target_sparsities = config.get('target_sparsities', None)
    
    # Use provided parameters or fall back to config values
    final_gate_hidden_size = gate_hidden_size or config_gate_hidden_size or 32
    final_target_sparsities = target_sparsities or config_target_sparsities or [0.5] * 16
    
    print(f"Creating SkippingMobileNetV3 with gate params: hidden_size={final_gate_hidden_size}")
    print(f"Target sparsities array (first 10): {final_target_sparsities[:10]}")
    print(f"Number of non-zero targets (gates to create): {sum(1 for ts in final_target_sparsities if ts != 0)}")
    
    # Get the building blocks from the original subnet
    first_conv = subnet.first_conv
    blocks = subnet.blocks
    
    # Create SkippingMobileNetV3 backbone with target sparsities array
    backbone = SkippingMobileNetV3(
        first_conv=first_conv, 
        blocks=blocks, 
        depth=depth,
        gate_type=gate_type,
        temperature=temperature,
        n_classes=n_classes,  # Add classifier for standalone use
        enable_gates=enable_gates,
        gate_hidden_size=final_gate_hidden_size,
        target_sparsities=final_target_sparsities
    )
    
    return backbone