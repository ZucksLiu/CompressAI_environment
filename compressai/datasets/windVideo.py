import random

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class WorldTensorDataset(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        tensors,
        specific_index,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")
        self.tensors = tensors
        self.specific_index = specific_index
        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # print(index)
        # sample_folder = self.sample_folders[3228]
        # print(self.sample_folders[3228])
        # samples = sorted(f for f in sample_folder.iterdir() if f.is_file())
        # print(samples)
        # exit()

        # sample_folder = self.tensors[index]
        # samples = sorted(f for f in sample_folder.iterdir() if f.is_file())


        # for p in frame_paths:
        #     z = np.asarray(Image.open(p).convert("RGB"))
        #     print(z.shape)
        # frames = np.stack([self.tensors[index+j] for j in range(3)], axis = 0)

        # frames = np.concatenate(
        #     [self.tensors[index+j] for j in range(3)], axis=-1
        # )
        real_index = self.specific_index[index]
        # print(self.specific_index.shape)
        # print(real_index)
        frames = [self.transform(self.tensors[real_index+12 * j]) for j in range(3)]
        # print(len(frames))
        # print(len(frames))
        # print(frames[0].shape)
        # frames = torch.chunk(self.transform(frames), 3)
        # print(frames.shape)
        # exit()
        if self.rnd_temp_order:
            if random.random() < 0.5:
                return frames[::-1]

        return frames

    def __len__(self):
        # print(self.specific_index)
        return np.size(self.specific_index)
        # return self.specific_index.size()
