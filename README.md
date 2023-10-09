## Swin Transformer PyTorch 

Repository to load Swin Transformers from image classification from PyTorch Hub with a single line import 

The official Swin transformer repository can be found here:

https://github.com/microsoft/Swin-Transformer

## Dependencies

- `torch` - PyTorch
- `timm` - Torchvision Image Models

## Load Model

```python
import torch
HUB_URL = "MaitreyaShelare/swin-transformer-model"
MODEL_NAME = "swin_tiny_patch4_window7_224"
# check hubconf for more models.
model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub
```

## Transforms

Transforms for passing in `PIL` images for inference.

```python
from torchvision import transforms as T
from PIL import Image
import timm

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
])
```
