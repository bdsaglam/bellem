"""Computer vision utilities"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/ml.vision.ipynb.

# %% auto 0
__all__ = ['Make3Channel', 'TorchVisionTransform']

# %% ../../nbs/ml.vision.ipynb 3
from PIL.Image import Image 
from torchvision.transforms import Compose

from fastcore.transform import Transform
from fastai.vision.core import PILImage

# %% ../../nbs/ml.vision.ipynb 4
class Make3Channel:
    """Tiles 1 channel image to 3 channel"""

    def __call__(self, x):
        if isinstance(x, Image):
            return x.convert(mode='RGB')
        rpts = (3, 1, 1) if x.ndim == 3 else (1, 3, 1, 1)
        return x.repeat(*rpts)
    
    def __repr__(self):
        return "Make3Channel()"

# %% ../../nbs/ml.vision.ipynb 5
class TorchVisionTransform(Transform):
    """Converts a torchvision transform to fastai transform"""
    def __init__(self, transform):
        self.tfm = transform

    def encodes(self, o: PILImage): 
        return self.tfm(o)
