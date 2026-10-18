"""
A package for common utility functions in 3D computer graphics and vision. Providing NumPy utilities in `utils3d.numpy`, PyTorch utilities in `utils3d.torch`, and IO utilities in `utils3d.io`.
"""
import importlib
from typing import TYPE_CHECKING

try:
    from ._unified import *
except ImportError:
    pass

__all__ = ['numpy', 'torch', 'io', 'pt']

def __getattr__(name: str):
    # 'pt' is an alias for 'torch', compatible with MoGe v2
    if name == 'pt':
        return importlib.import_module('.torch', __package__)
    return globals().get(name, importlib.import_module(f'.{name}', __package__)) 

if TYPE_CHECKING:
    from . import torch
    from . import torch as pt  # alias for MoGe v2 compatibility
    from . import numpy
    from . import io