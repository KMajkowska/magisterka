import numpy as np
from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    ToTensord,
)


def get_transforms(patch_size: Tuple[int, int, int]):
    """Tworzy pipeline transformacji dla danych treningowych i walidacyjnych."""
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # Sampler wycinający wycinki z 75% szansą na obecność ogniska chorobowego
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,
            neg=1,
            num_samples=2,
            image_key="image",
        ),
        # Augmentacja przestrzenna
        RandRotated(
            keys=["image", "label"],
            range_x=np.pi / 12, range_y=np.pi / 12, range_z=np.pi / 12,
            prob=0.5, mode=["bilinear", "nearest"]
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9, max_zoom=1.1,
            prob=0.3, mode=["bilinear", "nearest"]
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.1),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms
