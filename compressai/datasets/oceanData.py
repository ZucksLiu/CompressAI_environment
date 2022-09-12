from torch.utils.data import Dataset
import torch
import random
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms=None, prob=None, get_index=True, dataset_index=1):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms
        self.prob = prob
        self.get_index = get_index
        self.dataset_index = dataset_index

    def __getitem__(self, index):
        x = self.tensors[index]
        # torch.isinf(x)
        # print(x)
        # print(x.shape)
        if self.prob:
            k = random.choices(self.transforms, weights=self.prob)[0]
            if self.transforms:
                # print(k)
                x = k(x)
                # print(x)
        else:
            if self.transforms:
                x = self.transforms(x)
        if self.get_index == False:
            return x
        else:
            # print(index.shape)
            # sleep
            return x, index, self.dataset_index

    def __len__(self):
        return self.tensors.size(0)